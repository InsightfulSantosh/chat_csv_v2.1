import pandas as pd
from typing import Dict, List, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.workflow_nodes import query_df_node, create_plot_node
from utils.llm_factory import LLMFactory
from utils.base_dataframe_manager import BaseDataFrameManager
from utils.query_analyzer import QueryAnalyzer
from utils.query_executor import format_output, QueryExecutor
from utils.plot_manager import PlotManager
from utils.context_filter import ContextFilter, FilterState, FuzzyMatcher

logger = setup_logger(__name__)

class AgentState(TypedDict):
    """State for the conditional LangGraph agent."""
    full_df: pd.DataFrame
    context_filter_state: Dict[str, Any]  # {'filters': ..., 'entity_memory': ...}
    filtered_df: pd.DataFrame
    query: str
    messages: Annotated[List, "The messages in the conversation"]
    query_analysis: Annotated[Dict, "Analysis of the user query"]
    current_tool: Annotated[str, "Current tool being used"]
    tool_result: Annotated[str, "Result from tool execution"]
    should_plot: Annotated[bool, "Whether plotting is allowed"]
    confidence: Annotated[float, "Confidence in the decision"]
    conversation_context: Dict[str, Any]  # Stores last result columns/values for dynamic context
    last_result_indices: Any  # Store indices of last filtered DataFrame
    is_complex: Annotated[bool, "Whether the query is complex"]
    sub_queries: Annotated[List[str], "List of sub-queries for complex queries"]

class ConditionalLangGraphAgent(BaseDataFrameManager):
    """Enhanced LangGraph agent with conditional workflow for intelligent tool selection."""
    
    def __init__(self, df: pd.DataFrame, llm=None, provider: str = None, model: str = None, temperature: float = None):
        super().__init__(df)
        self.fuzzy_matcher = FuzzyMatcher(self.df)
        self.query_analyzer = QueryAnalyzer(self.df)
        self.query_executor = QueryExecutor(llm, self.fuzzy_matcher, self.df, PlotManager(self.fuzzy_matcher))
        self.plot_manager = PlotManager(self.fuzzy_matcher)
        self.last_conversation_context = {}
        
        config = get_config_manager()
        self.provider = provider or config.llm_config.default_provider
        self.model = model or config.llm_config.default_model
        self.temperature = temperature or config.llm_config.default_temperature
        
        if llm is not None:
            self.llm = llm
            logger.info(f"🔧 Using provided LLM instance for {self.provider.upper()}")
        else:
            self.llm = LLMFactory.create_llm(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature
            )
            logger.info(f"🔧 Created new LLM instance for {self.provider.upper()}")
        
        self._build_conditional_workflow()
    
    def debug_context_state(self, state: AgentState) -> None:
        """Debug method to understand context filtering behavior."""
        query = state["query"]
        full_df = state["full_df"]
        filtered_df = state["filtered_df"]
        conversation_context = state.get("conversation_context", {})
        context_filter_state = state.get("context_filter_state", {})
        
        logger.info("🔍 DEBUG CONTEXT STATE:")
        logger.info(f"  Query: {query}")
        logger.info(f"  Full DF shape: {full_df.shape}")
        logger.info(f"  Filtered DF shape: {filtered_df.shape}")
        logger.info(f"  Active filters: {context_filter_state.get('filters', {})}")
        logger.info(f"  Last query context: {conversation_context.get('last_query_context', {})}")
        
        if "profession_category" in filtered_df.columns:
            unique_categories = filtered_df["profession_category"].nunique()
            logger.info(f"  Unique profession categories in filtered data: {unique_categories}")
        
        if "remote_work" in filtered_df.columns or "work_type" in filtered_df.columns:
            # Adjust column name based on your actual data structure
            remote_col = "remote_work" if "remote_work" in filtered_df.columns else "work_type"
            remote_count = len(filtered_df[filtered_df[remote_col].str.contains("remote", case=False, na=False)])
            logger.info(f"  Remote workers in filtered data: {remote_count}")

    def _is_followup_query(self, query: str) -> bool:
        """Check if this is a follow-up query referencing previous results."""
        followup_indicators = [
            "of them", "of those", "among them", "from them",
            "how many of", "which of", "what about", "from these",
            "within these", "in these"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in followup_indicators)

    def _is_reset_query(self, query: str) -> bool:
        """Helper to check if the query is a reset command."""
        reset_indicators = ["reset", "clear", "start over", "new query"]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in reset_indicators)
    
    def _apply_context_filter_node(self, state: AgentState) -> AgentState:
        """Apply context/entity memory filtering to the full DataFrame using the query and explicit filter-based context logic, supporting follow-up queries on row subsets."""
        query = state["query"]
        full_df = state["full_df"]
        context_filter_state = state.get("context_filter_state", {"filters": {}, "entity_memory": {}})
        conversation_context = state.get("conversation_context", {})
        last_result_indices = state.get("last_result_indices", None)

        fuzzy_matcher = self.fuzzy_matcher

        # For follow-up queries, use previous indices as the base DataFrame
        is_followup = False
        if hasattr(self, '_is_followup_query') and self._is_followup_query(query):
            is_followup = True
        elif hasattr(self, 'query_analyzer') and hasattr(self.query_analyzer, 'is_followup_query') and self.query_analyzer.is_followup_query(query):
            is_followup = True
        else:
            # Fallback: check for referent words
            from utils.context_filter import QueryProcessor
            config = get_config_manager()
            qp = QueryProcessor(config)
            if qp.has_referent_words(query):
                is_followup = True

        # Subset DataFrame if follow-up and indices are available
        if is_followup and last_result_indices is not None and len(last_result_indices) > 0:
            logger.info(f"🔗 Follow-up query detected: using previous row indices for context ({len(last_result_indices)} rows)")
            base_df = full_df.loc[last_result_indices].copy()
        else:
            base_df = full_df.copy()

        context_filter = ContextFilter(base_df, fuzzy_matcher)
        context_filter._state = FilterState(
            filters=context_filter_state.get("filters", {}).copy(),
            entity_memory=context_filter_state.get("entity_memory", {}).copy()
        )

        # --- Explicit context logic ---
        if context_filter.query_processor.is_reset_command(query) or context_filter.query_processor.is_global_query(query):
            context_filter._state.filters = {}
            context_filter._state.entity_memory = {}
            logger.info("🔄 Reset/global query detected - clearing all filters")
        else:
            corrected_column = context_filter.query_processor.is_correction(query)
            if corrected_column:
                logger.info(f"✏️ Correction query detected for column: {corrected_column}")
                pass
            else:
                cleaned_query, query_words = context_filter.query_processor.clean_query(query)
                new_filters = context_filter._extract_filters(query_words, query)
                if new_filters:
                    logger.info(f"🆕 New filters extracted: {new_filters}")
                    
                    # Merge new filters with existing filters, replacing if the key already exists
                    for key, value in new_filters.items():
                        context_filter._state.filters[key] = value

                    logger.info(f"Filters after merge: {context_filter._state.filters}")
                else:
                    logger.info("No new filters extracted from query.")

        # Apply the filters
        context_filter._apply_state(context_filter._state)
        filtered_df = context_filter.df.copy()
        filtered_indices = filtered_df.index.tolist()

        state["filtered_df"] = filtered_df
        state["context_filter_state"] = {
            "filters": context_filter._state.filters.copy(),
            "entity_memory": context_filter._state.entity_memory.copy()
        }
        state["last_result_indices"] = filtered_indices
        
        # Update conversation_context with the latest state
        conversation_context = state.get("conversation_context", {})
        conversation_context["context_filter_state"] = {
            "filters": context_filter._state.filters.copy(),
            "entity_memory": context_filter._state.entity_memory.copy()
        }
        conversation_context["last_result_indices"] = filtered_indices
        conversation_context["last_query"] = query
        conversation_context["filtered_row_count"] = len(filtered_df)
        
        state["conversation_context"] = conversation_context
        
        self.debug_context_state(state)
        return state
    
    def _build_conditional_workflow(self):
        logger.info("Building conditional LangGraph workflow.")
        workflow = StateGraph(AgentState)
        # Add context filter node as the first step
        workflow.add_node("apply_context_filter", self._apply_context_filter_node)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("data_query", self._execute_data_query_node)
        workflow.add_node("visualization", self._execute_visualization_node)
        workflow.add_node("complex_query_handler", self._execute_complex_query_node)
        workflow.add_node("format_response", self._format_response_node)
        # Routing: context filter -> analyze -> conditional
        workflow.add_edge("apply_context_filter", "analyze_query")
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "visualization": "visualization",
                "data_query": "data_query",
                "complex_query_handler": "complex_query_handler"
            }
        )
        workflow.add_edge("data_query", "format_response")
        workflow.add_edge("visualization", "format_response")
        workflow.add_edge("complex_query_handler", "format_response")
        workflow.add_edge("format_response", END)
        workflow.set_entry_point("apply_context_filter")
        self.graph = workflow.compile()
    
    def _analyze_query_node(self, state: AgentState) -> AgentState:
        user_query = state["query"]
        logger.info(f"🔍 Analyzing query: {user_query}")
        analysis = self.query_analyzer.analyze_query(user_query)
        logger.info(f"📊 Query analysis: {analysis}")
        state["query_analysis"] = analysis
        state["is_complex"] = analysis.get("is_complex", False)
        state["confidence"] = analysis.get("confidence", 0.0)
        if state["is_complex"]:
            state["sub_queries"] = analysis.get("sub_queries", [])
            state["current_tool"] = "complex_query_handler"
        else:
            state["should_plot"] = analysis.get("should_plot", False)
            state["confidence"] = analysis.get("confidence", 0.0)
            state["current_tool"] = analysis.get("suggested_tool", "query_df")
            if state["should_plot"]:
                state["plot_type"] = analysis.get("plot_type")
                state["x_column"] = analysis.get("x_column")
                state["y_column"] = analysis.get("y_column")
        return state

    def _route_query(self, state: AgentState) -> str:
        if state.get("is_complex"):
            logger.info("🔄 Routing to complex query handler")
            return "complex_query_handler"
        
        should_plot = state.get("should_plot", False)
        confidence = state.get("confidence", 0.0)
        logger.info(f"🔄 Routing decision: should_plot={should_plot}, confidence={confidence}")
        if should_plot and confidence >= 0.7:
            return "visualization"
        else:
            return "data_query"
    
    def _execute_data_query_node(self, state: AgentState) -> AgentState:
        logger.info(f"📊 Executing data query node")
        try:
            # Use filtered_df from state
            filtered_df = state["filtered_df"]
            self.query_executor.update_dataframe(filtered_df)
            
            # Add context information to the query for better understanding
            query = state["query"]
            conversation_context = state.get("conversation_context", {})
            
            # For follow-up queries, add context to help the executor understand
            if self._is_followup_query(query):
                context_info = []
                last_query_context = conversation_context.get("last_query_context", {})
                
                if last_query_context:
                    for col, values in last_query_context.items():
                        if len(values) <= 20:  # Don't overwhelm with too many values
                            context_info.append(f"{col}: {', '.join(map(str, values))}")
                
                if context_info:
                    contextual_query = f"{query}\n\nContext from previous query: {'; '.join(context_info)}"
                    logger.info(f"📝 Enhanced query with context: {contextual_query}")
                    
                    # Temporarily update the query for execution
                    original_query = state["query"]
                    state["query"] = contextual_query
                    state = query_df_node(state, self.query_executor)
                    state["query"] = original_query  # Restore original
                else:
                    state = query_df_node(state, self.query_executor)
            else:
                state = query_df_node(state, self.query_executor)
            
            state["current_tool"] = "query_df"
            
            # Log filtering information for debugging
            total_rows = len(state["full_df"])
            filtered_rows = len(filtered_df)
            logger.info(f"✅ Data query executed: {filtered_rows}/{total_rows} rows after filtering")

            # After execution, check if the result should filter the dataframe for the next turn
            raw_result = state.get("raw_tool_result")
            if isinstance(raw_result, pd.Series):
                # This is likely a value_counts() or similar aggregation.
                # The index of the series contains the values to filter by.
                filter_values = raw_result.index.tolist()
                
                # Find the column that was aggregated. This is a bit of a heuristic.
                # We'll look for the column name in the query.
                query_lower = state["query"].lower()
                aggregated_col = None
                for col in state["full_df"].columns:
                    if col.lower() in query_lower:
                        aggregated_col = col
                        break
                
                if aggregated_col:
                    logger.info(f"Found aggregation on column '{aggregated_col}'. Filtering for next turn.")
                    # Filter the original dataframe by the values in the series index
                    filtered_df_for_next_turn = state["full_df"][state["full_df"][aggregated_col].isin(filter_values)]
                    new_indices = filtered_df_for_next_turn.index.tolist()
                    logger.info(f"New indices count: {len(new_indices)}")
                    state["last_result_indices"] = new_indices
                    
                    # Also update the conversation_context so it's persisted
                    state["conversation_context"]["last_result_indices"] = new_indices
                    
                    # Persist the filter that was just applied
                    if "context_filter_state" not in state["conversation_context"]:
                        state["conversation_context"]["context_filter_state"] = {"filters": {}}
                    state["conversation_context"]["context_filter_state"]["filters"][aggregated_col] = filter_values
                    logger.info(f"Updated conversation_context with {len(new_indices)} indices and filter on '{aggregated_col}'.")

        except Exception as e:
            error_msg = f"Error executing data query: {str(e)}"
            logger.error(error_msg)
            state["tool_result"] = error_msg
        
        return state

    def _execute_complex_query_node(self, state: AgentState) -> AgentState:
        logger.info("Executing complex query node")
        sub_queries = state.get("sub_queries", [])
        results = []
        all_indices = []

        for sub_query in sub_queries:
            logger.info(f"Executing sub-query: {sub_query}")
            
            # Create an isolated state for each sub-query
            sub_state = {
                "query": sub_query,
                "full_df": state["full_df"],
                "filtered_df": state["full_df"],  # Use the full_df for each sub-query
                "messages": [],
            }

            # Analyze the sub-query
            analysis = self.query_analyzer._analyze_simple_query(sub_query)
            sub_state["query_analysis"] = analysis
            
            # Execute the appropriate node based on the analysis
            if analysis.get("should_plot"):
                executed_state = self._execute_visualization_node(sub_state)
            else:
                executed_state = self._execute_data_query_node(sub_state)
            
            results.append(executed_state.get("tool_result", f"No result for sub-query: {sub_query}"))

            # Accumulate indices from the raw result
            raw_result = executed_state.get("raw_tool_result")
            if isinstance(raw_result, pd.DataFrame):
                all_indices.extend(raw_result.index.tolist())

        # Combine the results
        state["tool_result"] = "\n\n".join(results)
        
        # Update last_result_indices with the combined unique indices
        if all_indices:
            unique_indices = sorted(list(set(all_indices)))
            state["last_result_indices"] = unique_indices
            state["conversation_context"]["last_result_indices"] = unique_indices
            logger.info(f"Updated last_result_indices with {len(unique_indices)} combined indices from complex query.")

        return state

    def _execute_visualization_node(self, state: AgentState) -> AgentState:
        logger.info(f"📈 Executing visualization node")
        try:
            # Use filtered_df from state
            analysis = state.get("query_analysis", {})
            state['plot_type'] = analysis.get("plot_type")
            state['x_column'] = analysis.get("x_column")
            state['y_column'] = analysis.get("y_column")
            result_state = create_plot_node(state, self.plot_manager, state["filtered_df"])
            result_state["current_tool"] = "create_plot"
            logger.info(f"✅ Visualization executed successfully")
            return result_state
        except Exception as e:
            error_msg = f"Error executing visualization: {str(e)}"
            logger.error(error_msg)
            state["tool_result"] = error_msg
            return state
    
    def _format_response_node(self, state: AgentState) -> AgentState:
        tool_result = state.get("tool_result", "")
        current_tool = state.get("current_tool", "")
        confidence = state.get("confidence", 0.0)
        conversation_context = state.get("conversation_context", {})
        
        if current_tool == "create_plot":
            response = f"📈 Visualization created successfully!\n\n{tool_result}"
        else:
            response = f"📊 Data Analysis Result:\n\n{tool_result}"
        
        # Add context information for debugging if needed
        if confidence < 0.8:
            response += f"\n\n💡 Note: Confidence level: {confidence:.2f}"
        
        # Add filtering information for transparency
        filtered_count = conversation_context.get("filtered_row_count", 0)
        if filtered_count > 0:
            total_count = len(state["full_df"])
            if filtered_count < total_count:
                response += f"\n\n📊 Analysis based on {filtered_count} filtered records (from {total_count} total)"
        
        # Log the state of last_result_indices before finishing
        final_indices = conversation_context.get("last_result_indices", [])
        logger.info(f"📝 CONTEXT FOR NEXT TURN: last_result_indices contains {len(final_indices)} indices.")

        state["messages"].append(AIMessage(content=response))
        return state
    
    def invoke(self, query: str, thread_id: str = "default", output_format: str = None, conversation_context: Dict[str, Any] = None) -> str:
        try:
            logger.info(f"🚀 Invoking conditional workflow agent - Thread: {thread_id}")
            
            # Use conversation_context to maintain state
            conversation_context = conversation_context or {}
            context_filter_state = conversation_context.get("context_filter_state", {"filters": {}, "entity_memory": {}})
            last_result_indices = conversation_context.get("last_result_indices", None)

            initial_state = {
                "full_df": self.df.copy(),
                "context_filter_state": context_filter_state,
                "filtered_df": self.df.copy(),
                "query": query,
                "messages": [HumanMessage(content=query)],
                "query_analysis": {},
                "current_tool": "",
                "tool_result": "",
                "should_plot": False,
                "confidence": 0.0,
                "output_format": output_format or "table",
                "conversation_context": conversation_context,
                "last_result_indices": last_result_indices
            }
            final_state = self.graph.invoke(initial_state)
            final_message = final_state["messages"][-1]
            
            # The conversation_context is updated in place, so we don't need to return it.
            # self.last_conversation_context is updated by reference.
            self.last_conversation_context = final_state.get("conversation_context", {})

            if isinstance(final_message, AIMessage):
                return final_message.content
            else:
                return str(final_message)
        except Exception as e:
            logger.error(f"Error invoking conditional workflow agent: {str(e)}")
            return f"Error: {str(e)}"
    
    def _on_dataframe_update(self, new_df: pd.DataFrame) -> None:
        self.df = new_df.copy()
        self.fuzzy_matcher = FuzzyMatcher(self.df)
        self.query_executor.update_dataframe(new_df)
        self.query_executor.fuzzy_matcher = self.fuzzy_matcher
        self.query_executor.code_fixer.fuzzy_matcher = self.fuzzy_matcher
        self.plot_manager = PlotManager(self.fuzzy_matcher)
        self._build_conditional_workflow()
    
    def get_workflow_info(self) -> Dict[str, Any]:
        return {
            "workflow_type": "conditional",
            "nodes": ["apply_context_filter", "analyze_query", "data_query", "visualization", "format_response"],
            "conditional_edges": ["should_route_to_visualization"],
            "provider": self.provider,
            "model": self.model
        }
