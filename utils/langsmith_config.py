"""
LangSmith configuration and utilities for SmartPandasAgent.

This module provides LangSmith integration for observability, debugging, and monitoring
of the SmartPandasAgent's LLM interactions.
"""

import os
import re
from typing import Optional, Dict, Any
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils.logger import setup_logger
from utils.config_manager import get_config_manager

logger = setup_logger(__name__)

class LangSmithManager:
    """Manages LangSmith integration for observability and debugging."""
    
    def __init__(self):
        self.config = get_config_manager()
        self.client = None
        self.tracer = None
        self.is_enabled = False
        self._setup_langsmith()
    
    def _setup_langsmith(self):
        """Setup LangSmith configuration."""
        # Check if LangSmith is enabled
        self.is_enabled = self.config.langsmith_config.enabled
        
        if not self.is_enabled:
            logger.info("🔕 LangSmith is disabled. Set LANGCHAIN_TRACING_V2=true to enable.")
            return
        
        # Get LangSmith credentials
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project_name = os.getenv("LANGCHAIN_PROJECT", "smart-pandas-agent")
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        if not api_key:
            logger.warning("⚠️ LANGCHAIN_API_KEY not found. LangSmith will be disabled.")
            self.is_enabled = False
            return
        
        try:
            # Initialize LangSmith client
            self.client = Client(
                api_url=endpoint,
                api_key=api_key
            )
            
            # Set environment variables for LangChain
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = project_name
            os.environ["LANGCHAIN_API_KEY"] = api_key
            
            # Initialize tracer
            self.tracer = LangChainTracer(
                project_name=project_name
            )
            
            logger.info(f"✅ LangSmith initialized successfully - Project: {project_name}")
            logger.info(f"🔗 LangSmith URL: https://smith.langchain.com/project/{project_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangSmith: {e}")
            self.is_enabled = False
    
    def get_callback_manager(self, streaming: bool = False) -> Optional[CallbackManager]:
        """Get callback manager with LangSmith tracing."""
        if not self.is_enabled or not self.tracer:
            return None
        
        callbacks = [self.tracer]
        
        if streaming:
            callbacks.append(StreamingStdOutCallbackHandler())
        
        return CallbackManager(callbacks)
    
    def create_run_name(self, query: str, provider: str, model: str) -> str:
        """Create a descriptive run name for LangSmith."""
        # Truncate query for readability
        query_preview = query[:50] + "..." if len(query) > 50 else query
        return f"{provider}-{model}: {query_preview}"
    
    def log_metadata(self, run_id: str, metadata: Dict[str, Any]):
        """Log additional metadata to a LangSmith run."""
        if not self.is_enabled or not self.client:
            return
        
        # Check if run_id is a valid UUID format
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        
        if not uuid_pattern.match(run_id):
            # If run_id is not a valid UUID, log it as a custom event instead
            logger.debug(f"📊 Logging custom metadata event: {run_id}")
            logger.debug(f"📊 Metadata: {metadata}")
            return
        
        try:
            self.client.update_run(
                run_id=run_id,
                extra=metadata
            )
            logger.debug(f"📊 Logged metadata to run {run_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log metadata: {e}")
    
    def get_run_url(self, run_id: str) -> str:
        """Get the URL for a specific run in LangSmith."""
        if not self.is_enabled:
            return "LangSmith not enabled"
        
        project_name = os.getenv("LANGCHAIN_PROJECT", "smart-pandas-agent")
        return f"https://smith.langchain.com/project/{project_name}/runs/{run_id}"
    
    def list_recent_runs(self, limit: int = 10):
        """List recent runs from LangSmith for the current project.
        
        Args:
            limit: Maximum number of runs to return (default: 10)
            
        Returns:
            List of run objects from LangSmith
        """
        if not self.is_enabled or not self.client:
            logger.warning("LangSmith not enabled")
            return []
        
        try:
            # Get the current project name
            project_name = os.getenv("LANGCHAIN_PROJECT", "smart-pandas-agent")
            
            # List runs for the current project
            runs = list(self.client.list_runs(
                project_name=project_name,
                limit=limit
            ))
            logger.debug(f"Retrieved {len(runs)} runs from project: {project_name}")
            return runs
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []
    
    def get_run_details(self, run_id: str):
        """Get detailed information about a specific run."""
        if not self.is_enabled or not self.client:
            logger.warning("LangSmith not enabled")
            return None
        
        try:
            run = self.client.read_run(run_id)
            return run
        except Exception as e:
            logger.error(f"Failed to get run details: {e}")
            return None
    
    def list_runs_by_session(self, session_id: str, limit: int = 10):
        """List runs for a specific session."""
        if not self.is_enabled or not self.client:
            logger.warning("LangSmith not enabled")
            return []
        
        try:
            runs = list(self.client.list_runs(
                session_id=session_id,
                limit=limit
            ))
            return runs
        except Exception as e:
            logger.error(f"Failed to list runs by session: {e}")
            return []
    
    def list_runs_by_trace(self, trace_id: str, limit: int = 10):
        """List runs for a specific trace."""
        if not self.is_enabled or not self.client:
            logger.warning("LangSmith not enabled")
            return []
        
        try:
            runs = list(self.client.list_runs(
                trace_id=trace_id,
                limit=limit
            ))
            return runs
        except Exception as e:
            logger.error(f"Failed to list runs by trace: {e}")
            return []
    
    def log_custom_event(self, event_type: str, metadata: Dict[str, Any]):
        """Log a custom event without requiring a run ID."""
        if not self.is_enabled:
            logger.debug(f"📊 Custom event (LangSmith disabled): {event_type}")
            logger.debug(f"📊 Metadata: {metadata}")
            return
        
        logger.debug(f"📊 Custom event logged: {event_type}")
        logger.debug(f"📊 Metadata: {metadata}")
        
        # For now, just log to debug. In the future, this could be extended
        # to create custom runs or use LangSmith's event system

# Global LangSmith manager instance
langsmith_manager = LangSmithManager()

def get_langsmith_manager() -> LangSmithManager:
    """Get the global LangSmith manager instance."""
    return langsmith_manager 