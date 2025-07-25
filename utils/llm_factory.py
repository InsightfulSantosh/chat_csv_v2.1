"""
LLM Factory for creating flexible LLM instances across different providers.
Supports Anthropic, Google, and OpenAI providers with LangSmith integration.
"""

from typing import Optional, Union
from langchain_core.language_models import BaseLLM
from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.provider_registry import get_provider_registry, create_llm as registry_create_llm
from utils.langsmith_config import get_langsmith_manager

logger = setup_logger(__name__)


class LLMFactory:
    """Factory class for creating LLM instances from different providers."""
    
    @staticmethod
    def create_llm(
        provider: str = None,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        with_langsmith: bool = True
    ) -> BaseLLM:
        """
        Create an LLM instance for the specified provider.
        
        Args:
            provider: LLM provider ('anthropic', 'google', 'openai')
            model: Model name for the provider
            temperature: Temperature for the model
            api_key: API key for the provider (optional, uses config if not provided)
            with_langsmith: Whether to include LangSmith callbacks
            
        Returns:
            LLM instance
        """
        config = get_config_manager()
        provider = provider or config.llm_config.default_provider
        model = model or config.llm_config.default_model
        temperature = temperature or config.llm_config.default_temperature
        
        # Get the correct model name for the provider
        model_name = config.get_model_name(provider, model)
        
        # Log LLM creation details
        logger.info(f"🤖 Creating LLM instance - Provider: {provider.upper()}, Model: {model_name}, Temperature: {temperature}")
        logger.info(f"📋 LLM Configuration - Original Model: {model}, Mapped Model: {model_name}")
        
        # Create the LLM instance
        llm = registry_create_llm(provider, model_name, temperature, api_key)
        
        # Add LangSmith callbacks if enabled
        if with_langsmith:
            langsmith_manager = get_langsmith_manager()
            if langsmith_manager.is_enabled:
                callback_manager = langsmith_manager.get_callback_manager()
                if callback_manager:
                    llm.callbacks = callback_manager
                    logger.info(f"🔗 LangSmith tracing enabled for {provider.upper()} LLM")
        
        return llm
    

    
    @staticmethod
    def get_available_models(provider: str = None) -> dict:
        """
        Get available models for the specified provider.
        
        Args:
            provider: Provider name ('anthropic', 'google', 'openai')
            
        Returns:
            Dictionary of available models
        """
        config = get_config_manager()
        return config.get_available_models(provider)
    
    @staticmethod
    def validate_provider_config(provider: str) -> bool:
        """
        Validate that the required API key is configured for the provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if configuration is valid
        """
        registry = get_provider_registry()
        return registry.validate_provider(provider)
    
    @staticmethod
    def get_available_providers() -> dict:
        """
        Get all available providers with their display names.
        
        Returns:
            Dictionary mapping provider names to display names
        """
        registry = get_provider_registry()
        return registry.get_available_providers()


 