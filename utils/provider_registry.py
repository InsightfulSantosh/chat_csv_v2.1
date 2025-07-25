"""
Provider Registry for LLM providers using the Factory Pattern.
Provides a more elegant and extensible way to manage different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any
from langchain_core.language_models import BaseLLM
from utils.config_manager import get_config_manager
from utils.logger import setup_logger

# Optional imports for different providers
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None

logger = setup_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_llm(self, model: str, temperature: float, api_key: str) -> BaseLLM:
        """Create an LLM instance for this provider."""
        # This method should be implemented by subclasses
    
    @abstractmethod
    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the API key for this provider."""
        # This method should be implemented by subclasses
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that the provider is properly configured."""
        # This method should be implemented by subclasses
    
    @abstractmethod
    def get_display_name(self) -> str:
        """Get the display name for this provider."""
        # This method should be implemented by subclasses


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation."""
    
    def create_llm(self, model: str, temperature: float, api_key: str) -> BaseLLM:
        """Create an Anthropic LLM instance."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("langchain_anthropic is not installed. Please install it with: pip install langchain-anthropic")
        
        logger.info(f"🔵 Anthropic LLM created successfully - Model: {model}, Temperature: {temperature}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key
        )
    
    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the Anthropic API key."""
        config = get_config_manager()
        return api_key or config.get_api_key('anthropic')
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        config = get_config_manager()
        return config.validate_provider_config('anthropic')
    
    def get_display_name(self) -> str:
        """Get the display name for Anthropic."""
        return "Anthropic"


class GoogleProvider(LLMProvider):
    """Google LLM provider implementation."""
    
    def create_llm(self, model: str, temperature: float, api_key: str) -> BaseLLM:
        """Create a Google LLM instance."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("langchain_google_genai is not installed. Please install it with: pip install langchain-google-genai")
        
        logger.info(f"🟢 Google LLM created successfully - Model: {model}, Temperature: {temperature}")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    
    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the Google API key."""
        config = get_config_manager()
        return api_key or config.get_api_key('google')
    
    def validate_config(self) -> bool:
        """Validate Google configuration."""
        config = get_config_manager()
        return config.validate_provider_config('google')
    
    def get_display_name(self) -> str:
        """Get the display name for Google."""
        return "Google"


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def create_llm(self, model: str, temperature: float, api_key: str) -> BaseLLM:
        """Create an OpenAI LLM instance."""
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain_openai is not installed. Please install it with: pip install langchain-openai")
        
        logger.info(f"🟣 OpenAI LLM created successfully - Model: {model}, Temperature: {temperature}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key
        )
    
    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Get the OpenAI API key."""
        config = get_config_manager()
        return api_key or config.get_api_key('openai')
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        config = get_config_manager()
        return config.validate_provider_config('openai')
    
    def get_display_name(self) -> str:
        """Get the display name for OpenAI."""
        return "OpenAI"


class ProviderRegistry:
    """Registry for managing LLM providers using the Factory Pattern."""
    
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register the default providers."""
        self.register_provider('anthropic', AnthropicProvider())
        self.register_provider('google', GoogleProvider())
        self.register_provider('openai', OpenAIProvider())
    
    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """
        Register a new provider.
        
        Args:
            name: Provider name (lowercase)
            provider: Provider instance
        """
        self._providers[name.lower()] = provider
        logger.info(f"✅ Registered provider: {name} ({provider.get_display_name()})")
    
    def get_provider(self, name: str) -> LLMProvider:
        """
        Get a provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider not found
        """
        provider = self._providers.get(name.lower())
        if not provider:
            available = ', '.join(self._providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available providers: {available}")
        return provider
    
    def get_available_providers(self) -> Dict[str, str]:
        """
        Get all available providers with their display names.
        
        Returns:
            Dictionary mapping provider names to display names
        """
        return {name: provider.get_display_name() 
                for name, provider in self._providers.items()}
    
    def validate_provider(self, name: str) -> bool:
        """
        Validate that a provider is properly configured.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is configured
        """
        try:
            provider = self.get_provider(name)
            return provider.validate_config()
        except ValueError:
            return False
    
    def create_llm(self, provider_name: str, model: str, temperature: float, 
                  api_key: Optional[str] = None) -> BaseLLM:
        """
        Create an LLM instance using the specified provider.
        
        Args:
            provider_name: Name of the provider
            model: Model name
            temperature: Temperature setting
            api_key: API key (optional)
            
        Returns:
            LLM instance
        """
        provider = self.get_provider(provider_name)
        api_key = provider.get_api_key(api_key)
        
        if not api_key:
            raise ValueError(f"{provider.get_display_name()} API key is required")
        
        return provider.create_llm(model, temperature, api_key)


# Global registry instance
_provider_registry = ProviderRegistry()


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    return _provider_registry


def register_provider(name: str, provider: LLMProvider) -> None:
    """Register a new provider in the global registry."""
    _provider_registry.register_provider(name, provider)


def get_provider(name: str) -> LLMProvider:
    """Get a provider from the global registry."""
    return _provider_registry.get_provider(name)


def create_llm(provider_name: str, model: str, temperature: float, 
              api_key: Optional[str] = None) -> BaseLLM:
    """Create an LLM instance using the global registry."""
    return _provider_registry.create_llm(provider_name, model, temperature, api_key) 