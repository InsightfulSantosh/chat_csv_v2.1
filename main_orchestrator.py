import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.fuzzy_matcher import FuzzyMatcher
from utils.plot_manager import PlotManager
from utils.prompts import PromptManager
from utils.langsmith_config import get_langsmith_manager
from agents.workflow_llm import LLMAgent
import json

logger = setup_logger(__name__)

class SmartPandasAgent:
    """Main SmartPandasAgent class that orchestrates all components. All context/entity memory filtering is now handled inside the LangGraph workflow."""
    
    def __init__(self, csv_path: str, provider: str = None, model: str = None, 
                 max_retries: int = None, temperature: float = None):
        """
        Initialize SmartPandasAgent with a CSV file path.
        Args:
            csv_path: Path to the CSV file
            provider: LLM provider (anthropic, google, openai)
            model: LLM model name
            max_retries: Maximum number of retry attempts
            temperature: LLM temperature setting
        """
        logger.info("🚀 Initializing SmartPandasAgent.")
        config = get_config_manager()
        self.provider = provider or config.llm_config.default_provider
        self.model = model or config.llm_config.default_model
        self.temperature = temperature or config.llm_config.default_temperature
        logger.info(f"🤖 LLM Configuration - Provider: {self.provider.upper()}, Model: {self.model}, Temperature: {self.temperature}")
        config.validate_config()
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV file not found at: {csv_path}")
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
        logger.info(f"📊 Loading CSV file: {csv_path}")
        self.df_full = pd.read_csv(csv_path)
        self.df = self.df_full.copy()
        logger.info(f"✅ Data loaded successfully - Shape: {self.df.shape}")
        logger.info("🔧 Initializing components...")
        self.fuzzy_matcher = FuzzyMatcher(self.df_full)
        self.plot_manager = PlotManager(self.fuzzy_matcher)
        self.prompt_manager = PromptManager()
        # Initialize LLM agent (all context/memory logic is now inside the workflow)
        logger.info(f"🤖 Initializing LLM Agent with {self.provider.upper()} provider...")
        self.llm_agent = LLMAgent(
            df=self.df,
            fuzzy_matcher=self.fuzzy_matcher,
            plot_manager=self.plot_manager,
            provider=self.provider,
            model=self.model,
            max_retries=max_retries or config.llm_config.max_retries,
            temperature=self.temperature
        )
        logger.info("✅ SmartPandasAgent initialization complete!")

    def get_plot(self):
        """Get the current plot figure."""
        return self.plot_manager.get_plot()

    def save_plot(self, filename: str, dpi: int = None):
        """Save the current plot to a file."""
        return self.plot_manager.save_plot(filename, dpi)

    def get_dataframe_info(self):
        """Get basic information about the current DataFrame."""
        return {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "sample_data": self.df.head().to_dict() if not self.df.empty else {}
        }
    
    def suggest_columns(self, query: str) -> List[str]:
        """Suggest column names based on query."""
        suggested_cols = self.fuzzy_matcher.suggest_columns(query)
        if suggested_cols:
            suggestion_prompt = self.prompt_manager.get_column_suggestion_prompt(
                query=query,
                available_columns=list(self.df.columns),
                suggested_columns=suggested_cols
            )
            logger.debug(f"Column suggestion prompt: {suggestion_prompt}")
        return suggested_cols

    def query(self, question: str, thread_id: str = "default", output_format: str = "plain"):
        """Process a natural language question. All context/entity memory filtering is handled inside the workflow."""
        logger.info(f"💬 Processing query with {self.provider.upper()} ({self.model}) - Thread: {thread_id}")
        logger.info(f"❓ User question: {question}")
        langsmith_manager = get_langsmith_manager()
        metadata = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "thread_id": thread_id,
            "dataframe_shape": self.df.shape,
            "dataframe_columns": list(self.df.columns),
            "original_question": question
        }
        try:
            result = self.llm_agent.invoke(question, thread_id, output_format=output_format)
            fig = plt.gcf()
            if fig and fig.get_axes() and any(ax.has_data() for ax in fig.get_axes()):
                metadata["has_plot"] = True
                if langsmith_manager.is_enabled:
                    langsmith_manager.log_custom_event("success", metadata)
                return result, fig
            metadata["has_plot"] = False
            if langsmith_manager.is_enabled:
                langsmith_manager.log_custom_event("success", metadata)
            return result, None
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            if langsmith_manager.is_enabled:
                metadata["status"] = "error"
                metadata["error_message"] = str(e)
                langsmith_manager.log_custom_event("error", metadata)
            return f"Error: {str(e)}", None
