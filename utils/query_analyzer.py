import json
from typing import Dict, Any, List
import pandas as pd
from utils.logger import setup_logger
from utils.llm_factory import LLMFactory
from utils.prompts import PromptManager

logger = setup_logger(__name__)

class QueryAnalyzer:
    """
    Analyzes user queries to determine the appropriate tool using an LLM.
    """

    def __init__(self, df: pd.DataFrame, llm=None):
        """
        Initializes the QueryAnalyzer with a DataFrame and an optional LLM instance.

        Args:
            df (pd.DataFrame): The DataFrame to provide context for column names.
            llm (optional): An existing LLM instance. If not provided, a new one will be created.
        """
        self.df = df
        self.prompt_manager = PromptManager()
        if llm:
            self.llm = llm
        else:
            self.llm = LLMFactory.create_llm()

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes a user query using an LLM to determine if it's a simple or complex query,
        and routes to the appropriate analysis method.

        Args:
            query (str): The user's query string.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results.
        """
        # For simplicity in this example, we'll use a keyword-based check to decide
        # if a query is complex. In a real-world scenario, a more sophisticated
        # check (e.g., another LLM call) might be used.
        complex_keywords = ['and', ',', 'also', 'then', 'vs']
        if any(keyword in query.lower() for keyword in complex_keywords):
            logger.debug(f"Complex query detected: {query}")
            return self._analyze_complex_query(query)
        else:
            logger.debug(f"Simple query detected: {query}")
            return self._analyze_simple_query(query)

    def _analyze_simple_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes a simple user query using an LLM to determine the appropriate tool.
        """
        prompt = self.prompt_manager.get_custom_prompt(
            'query_analysis',
            query=query,
            columns=', '.join(self.df.columns)
        )
        try:
            response = self.llm.invoke(prompt)
            response_content = response.content
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in the LLM response.")
            json_string = response_content[json_start:json_end]
            analysis_result = json.loads(json_string)
            if not all(k in analysis_result for k in ['should_plot', 'confidence', 'suggested_tool']):
                raise ValueError("LLM response missing required keys for simple query analysis.")
            
            # Rule-based override for "top", "bottom", etc.
            query_lower = query.lower()
            analysis_keywords = ["top", "bottom", "highest", "lowest"]
            visualization_keywords = ["plot", "chart", "graph", "visualize"]

            is_analysis_query = any(keyword in query_lower for keyword in analysis_keywords)
            is_visualization_query = any(keyword in query_lower for keyword in visualization_keywords)

            if is_analysis_query and not is_visualization_query:
                logger.debug("Override: Forcing to data query based on keywords.")
                analysis_result['should_plot'] = False
                analysis_result['suggested_tool'] = 'query_df'

            # Fallback logic for plot_type and x_column if not provided by LLM
            if analysis_result.get('should_plot'):
                # Fallback for plot_type
                if not analysis_result.get('plot_type'):
                    plot_types = ['pie', 'piechart', 'bar', 'line', 'scatter', 'histogram', 'box', 'heatmap', 'countplot']
                    for plot_type in plot_types:
                        if plot_type in query_lower:
                            analysis_result['plot_type'] = plot_type
                            logger.debug(f"Fallback: extracted plot_type '{plot_type}' from query.")
                            break
                
                # Fallback for x_column
                if not analysis_result.get('x_column'):
                    for col in self.df.columns:
                        if col.lower() in query_lower:
                            analysis_result['x_column'] = col
                            logger.debug(f"Fallback: extracted x_column '{col}' from query.")
                            break
            
            analysis_result['is_complex'] = False
            return analysis_result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing LLM response for simple query analysis: {e}")
            return self._default_error_response()

    def _analyze_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes a complex user query by breaking it down into sub-queries using an LLM.
        """
        prompt = self.prompt_manager.get_custom_prompt(
            'complex_query_parser',
            query=query
        )
        try:
            response = self.llm.invoke(prompt)
            response_content = response.content
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in the LLM response.")
            json_string = response_content[json_start:json_end]
            analysis_result = json.loads(json_string)
            if 'sub_queries' not in analysis_result or not isinstance(analysis_result['sub_queries'], list):
                raise ValueError("LLM response missing 'sub_queries' key or it's not a list.")
            
            return {
                'is_complex': True,
                'sub_queries': analysis_result['sub_queries'],
                'confidence': 0.9, # High confidence as it's a structured breakdown
                'suggested_tool': 'complex_query_handler' # A new tool/node to handle this
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing LLM response for complex query analysis: {e}")
            # If complex parsing fails, fall back to simple analysis
            return self._analyze_simple_query(query)

    def _default_error_response(self) -> Dict[str, Any]:
        """
        Returns a default error response when LLM analysis fails.
        """
        return {
            'should_plot': False,
            'confidence': 0.5,
            'reasoning': 'Error processing LLM response, defaulting to data query.',
            'suggested_tool': 'query_df',
            'is_complex': False
        }

    def is_plotting_allowed(self, query: str) -> bool:
        """
        Simple boolean check for whether plotting should be allowed.
        """
        analysis = self.analyze_query(query)
        return analysis.get('should_plot', False)
