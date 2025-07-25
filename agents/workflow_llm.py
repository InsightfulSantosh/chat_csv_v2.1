import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.fuzzy_matcher import FuzzyMatcher
from utils.plot_manager import PlotManager
from utils.query_executor import QueryExecutor
from utils.base_dataframe_manager import BaseDataFrameManager
from utils.llm_factory import LLMFactory
from agents.workflow_conditional import ConditionalLangGraphAgent

logger = setup_logger(__name__)

class LLMAgent(BaseDataFrameManager):
    """Handles LLM agent creation and management for the SmartPandasAgent."""
    
    def __init__(self, df: pd.DataFrame, fuzzy_matcher: FuzzyMatcher, plot_manager: PlotManager,
                 provider: str = None, model: str = None, max_retries: int = None, temperature: float = None):
        super().__init__(df)
        self.fuzzy_matcher = fuzzy_matcher
        self.plot_manager = plot_manager
        
        config = get_config_manager()
        self.provider = provider or config.llm_config.default_provider
        self.model = model or config.llm_config.default_model
        self.temperature = temperature or config.llm_config.default_temperature
        self.max_retries = max_retries or config.llm_config.max_retries
        
        self.conversation_states: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"🤖 Initializing LLM Agent with {self.provider.upper()} provider...")
        
        # Get API key for the provider
        api_key = config.get_api_key(self.provider)
        if not api_key:
            raise ValueError(f"API key not found for provider: {self.provider.upper()}")
        
        logger.info(f"🔧 Creating LLM instance via factory...")
        self.llm = LLMFactory.create_llm(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            api_key=api_key
        )
        
        logger.info("🔧 Initializing QueryExecutor...")
        self.query_executor = QueryExecutor(self.llm, fuzzy_matcher, self.df, self.plot_manager)
        
        logger.info("🔧 Initializing Conditional LangGraph Agent...")
        # Initialize conditional LangGraph agent with the same LLM instance
        self.conditional_agent = ConditionalLangGraphAgent(
            self.df,
            llm=self.llm,  # Pass the existing LLM instance
            provider=self.provider, model=self.model, temperature=self.temperature
        )
        
        # Expose the query analyzer for direct access
        self.query_analyzer = self.conditional_agent.query_analyzer
        
        logger.info("✅ LLM Agent initialization complete!")
    

    def invoke(self, query: str, thread_id: str = "default", output_format: str = None):
        """Invoke the conditional workflow agent with a query and output format."""
        logger.info(f"🎯 Invoking {self.provider.upper()} conditional agent ({self.model}) - Thread: {thread_id}")
        logger.debug(f"📝 Query: {query}")
        
        # Retrieve conversation context for the given thread_id
        conversation_context = self.conversation_states.get(thread_id, {})
        logger.info(f"CONTEXT BEFORE INVOKE: {len(conversation_context.get('last_result_indices', []))} indices")

        result = self.conditional_agent.invoke(
            query, 
            thread_id, 
            output_format=output_format,
            conversation_context=conversation_context
        )
        
        # Update conversation context
        self.conversation_states[thread_id] = self.conditional_agent.last_conversation_context
        logger.info(f"CONTEXT AFTER INVOKE: {len(self.conversation_states[thread_id].get('last_result_indices', []))} indices")
        
        logger.info(f"✅ {self.provider.upper()} conditional agent response received")
        return result

    def _on_dataframe_update(self, new_df: pd.DataFrame) -> None:
        """Update the dataframe and rebuild the agent"""
        self.query_executor.update_dataframe(new_df)
        self.conditional_agent.update_dataframe(new_df)

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow."""
        return self.conditional_agent.get_workflow_info()
