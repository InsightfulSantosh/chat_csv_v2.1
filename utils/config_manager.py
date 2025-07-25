"""
Configuration Manager for centralized configuration handling.
Provides validation, caching, and environment-specific settings.
"""

import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from dotenv import load_dotenv
from utils.logger_mixin import LoggerMixin


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    default_provider: str = "google"
    default_model: str = "gemini-1.5-pro"
    default_temperature: float = 0.0
    max_retries: int = 2
    retry_delay: float = 0.5
    
    # Model mappings
    model_mappings: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'anthropic': {
            'claude-3-5-sonnet-20240620': 'claude-3-5-sonnet-20240620',
            'claude-3-opus-20240229': 'claude-3-opus-20240229',
            'claude-3-sonnet-20240229': 'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307': 'claude-3-haiku-20240307'
        },
        'google': {
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-pro': 'gemini-pro',
            'gemini-2.5-pro': 'gemini-2.5-pro'
        },
        'openai': {
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4-turbo': 'gpt-4-turbo-preview',
            'gpt-3.5-turbo': 'gpt-3.5-turbo'
        }
    })


@dataclass
class PlotConfig:
    """Configuration for plotting settings."""
    default_plot_size: tuple = (10, 6)
    default_dpi: int = 300
    plot_style: str = 'default'
    seaborn_palette: str = "husl"
    histogram_alpha: float = 0.7
    x_axis_label_rotation: int = 45
    heatmap_float_format: str = '.2f'
    pie_percent_format: str = '%1.1f%%'
    auto_plot_enabled: bool = False
    histogram_bins: int = 30


@dataclass
class DataConfig:
    """Configuration for data processing settings."""
    fuzzy_match_threshold: float = 0.75
    entity_memory_match_threshold: float = 0.8
    cardinality_ratio_threshold: float = 0.15
    max_categorical_values: int = 35
    max_categorical_examples: int = 10
    min_categorical_values_for_memory: int = 2
    max_categorical_values_for_memory: int = 100
    # Stopwords for entity matching (configurable)
    stopwords: list = field(default_factory=lambda: [
        "as", "in", "the", "of", "for", "on", "at", "by", "with", "to", "from", "and", "or", "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did", "a", "an", "it", "this", "that", "these", "those", "but", "if", "then", "so", "not", "no", "yes", "can", "will", "just", "now", "out", "up", "down", "off", "over", "under", "again", "more", "less", "very", "much", "some", "any", "all", "each", "every", "either", "neither", "both", "which", "who", "whom", "whose", "what", "when", "where", "why", "how"
    ])
    # Generic term detection settings
    enable_generic_term_detection: bool = True
    generic_term_threshold: int = 2  # Minimum matches to consider a term generic
    # Output formatting settings
    default_output_format: str = "auto"  # "auto", "table", "string", "summary"
    max_table_rows: int = 50
    max_table_width: int = 20
    show_dataframe_info: bool = True
    include_row_count: bool = True


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    # High-risk patterns that are always blocked (even in staging)
    forbidden_code_patterns: list = field(default_factory=lambda: [
        "exec", "eval", "subprocess", "os.system", "os.popen", 
        "__import__", "input(", "raw_input"
    ])
    
    # Medium-risk patterns that can be relaxed in staging
    staging_allowed_patterns: list = field(default_factory=lambda: [
        "import", "open(", "os.", "sys.", "__", "file", "with open"
    ])
    
    # Environment-based security level
    security_level: str = "staging"  # "production", "staging", "development"


@dataclass
class ContextConfig:
    """Configuration for context filtering settings."""
    referent_words: list = field(default_factory=lambda: [
        "these", "those", "them", "above", "that", "there", "their"
    ])
    correction_patterns: list = field(default_factory=lambda: [
        r"^it is (.+)$",
        r"^use (.+)$",
        r"^try (.+)$",
        r"^column is (.+)$",
        r"^it's (.+)$"
    ])
    global_query_keywords: list = field(default_factory=lambda: [
        "entire dataset", "overall", "whole data", "across all", 
        "full data", "without filters", "all data"
    ])
    reset_commands: list = field(default_factory=lambda: [
        "reset", "clear filters", "start over", "clear memory"
    ])


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(message)s"


@dataclass
class LangSmithConfig:
    """Configuration for LangSmith observability settings."""
    enabled: bool = False
    project_name: str = "smart-pandas-agent"
    endpoint: str = "https://api.smith.langchain.com"
    trace_verbose: bool = False
    log_metadata: bool = True
    log_inputs: bool = True
    log_outputs: bool = True


class ConfigManager(LoggerMixin):
    """
    Centralized configuration manager with validation and caching.
    
    This class provides a single point of access to all configuration
    settings with validation, caching, and environment-specific overrides.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        super().__init__()
        self._config_cache: Dict[str, Any] = {}
        self._config_file = config_file
        self._load_environment()
        self._load_config_file()
        self._initialize_configs()
        self._validate_config()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()
        self.log_info("Environment variables loaded")
    
    def _load_config_file(self) -> None:
        """Load configuration from file if specified."""
        if self._config_file and Path(self._config_file).exists():
            try:
                with open(self._config_file, 'r') as f:
                    file_config = json.load(f)
                self._config_cache.update(file_config)
                self.log_info(f"Configuration loaded from {self._config_file}")
            except Exception as e:
                self.log_warning(f"Failed to load config file {self._config_file}: {e}")
    
    def _initialize_configs(self) -> None:
        """Initialize all configuration sections."""
        # LLM Configuration
        self.llm_config = LLMConfig(
            default_provider=os.getenv('DEFAULT_PROVIDER', 'google'),
            default_model=os.getenv('DEFAULT_MODEL', 'gemini-1.5-pro'),
            default_temperature=float(os.getenv('DEFAULT_TEMPERATURE', 0)),
            max_retries=int(os.getenv('MAX_RETRIES', 2)),
            retry_delay=float(os.getenv('RETRY_DELAY', 0.5))
        )
        
        # Plot Configuration
        self.plot_config = PlotConfig(
            auto_plot_enabled=os.getenv('AUTO_PLOT_ENABLED', 'false').lower() == 'true'
        )
        
        # Data Configuration
        self.data_config = DataConfig(
            fuzzy_match_threshold=float(os.getenv('FUZZY_MATCH_THRESHOLD', 0.75)),
            entity_memory_match_threshold=float(os.getenv('ENTITY_MEMORY_MATCH_THRESHOLD', 0.8)),
            cardinality_ratio_threshold=float(os.getenv('CARDINALITY_RATIO_THRESHOLD', 0.15)),
            max_categorical_values=int(os.getenv('MAX_CATEGORICAL_VALUES', 35)),
            max_categorical_examples=int(os.getenv('MAX_CATEGORICAL_EXAMPLES', 10)),
            min_categorical_values_for_memory=int(os.getenv('MIN_CATEGORICAL_VALUES_FOR_MEMORY', 2)),
            max_categorical_values_for_memory=int(os.getenv('MAX_CATEGORICAL_VALUES_FOR_MEMORY', 100)),
            stopwords=os.getenv('STOPWORDS', 'as,in,the,of,for,on,at,by,with,to,from,and,or,is,are,was,were,be,been,has,have,had,do,does,did,a,an,it,this,that,these,those,but,if,then,so,not,no,yes,can,will,just,now,out,up,down,off,over,under,again,more,less,very,much,some,any,all,each,every,either,neither,both,which,who,whom,whose,what,when,where,why,how').split(','),
            enable_generic_term_detection=os.getenv('ENABLE_GENERIC_TERM_DETECTION', 'true').lower() == 'true',
            generic_term_threshold=int(os.getenv('GENERIC_TERM_THRESHOLD', 2)),
            default_output_format=os.getenv('DEFAULT_OUTPUT_FORMAT', 'auto'),
            max_table_rows=int(os.getenv('MAX_TABLE_ROWS', 20)),
            max_table_width=int(os.getenv('MAX_TABLE_WIDTH', 150)),
            show_dataframe_info=os.getenv('SHOW_DATAFRAME_INFO', 'true').lower() == 'true',
            include_row_count=os.getenv('INCLUDE_ROW_COUNT', 'true').lower() == 'true'
        )
        
        # Security Configuration
        security_level = os.getenv('SECURITY_LEVEL', 'production').lower()
        self.security_config = SecurityConfig(security_level=security_level)
        
        # Context Configuration
        self.context_config = ContextConfig()
        
        # Logging Configuration
        self.logging_config = LoggingConfig(
            log_level=os.getenv('LOG_LEVEL', "INFO")
        )
        
        # LangSmith Configuration
        self.langsmith_config = LangSmithConfig(
            enabled=os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
            project_name=os.getenv('LANGCHAIN_PROJECT', 'smart-pandas-agent'),
            endpoint=os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com'),
            trace_verbose=os.getenv('LANGCHAIN_TRACE_VERBOSE', 'false').lower() == 'true',
            log_metadata=os.getenv('LANGSMITH_LOG_METADATA', 'true').lower() == 'true',
            log_inputs=os.getenv('LANGSMITH_LOG_INPUTS', 'true').lower() == 'true',
            log_outputs=os.getenv('LANGSMITH_LOG_OUTPUTS', 'true').lower() == 'true'
        )
        
        # API Keys
        self.api_keys = {
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY')
        }
        
        self.log_configuration({
            'default_provider': self.llm_config.default_provider,
            'default_model': self.llm_config.default_model,
            'max_retries': self.llm_config.max_retries,
            'log_level': self.logging_config.log_level
        })
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        provider = self.llm_config.default_provider.lower()
        
        if provider == 'anthropic' and not self.api_keys['anthropic']:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found. Please set it in your .env file.")
        elif provider == 'google' and not self.api_keys['google']:
            raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file.")
        elif provider == 'openai' and not self.api_keys['openai']:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
        
        self.log_success("Configuration validation")
    
    def validate_config(self) -> None:
        """Public method to validate the configuration."""
        self._validate_config()
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for the specified provider.
        
        Args:
            provider: Provider name
            
        Returns:
            API key or None if not found
        """
        return self.api_keys.get(provider.lower())
    
    def get_model_name(self, provider: str = None, model: str = None) -> str:
        """
        Get the correct model name for the specified provider.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            Mapped model name
        """
        provider = provider or self.llm_config.default_provider
        model = model or self.llm_config.default_model
        
        # If no specific model is provided, use provider's default
        if not model or model == self.llm_config.default_model:
            default_models = {
                'anthropic': 'claude-3-5-sonnet-20240620',
                'google': 'gemini-1.5-pro',
                'openai': 'gpt-4o'
            }
            model = default_models.get(provider, self.llm_config.default_model)
        
        if provider in self.llm_config.model_mappings and model in self.llm_config.model_mappings[provider]:
            return self.llm_config.model_mappings[provider][model]
        
        return model
    
    def validate_provider_config(self, provider: str) -> bool:
        """
        Validate that a provider is properly configured.
        
        Args:
            provider: Provider name
            
        Returns:
            True if provider is configured
        """
        return bool(self.get_api_key(provider))
    
    def get_available_models(self, provider: str = None) -> Dict[str, str]:
        """
        Get available models for the specified provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary of available models
        """
        if provider:
            return self.llm_config.model_mappings.get(provider.lower(), {})
        return self.llm_config.model_mappings
    
    def get_config_section(self, section: str) -> Any:
        """
        Get a configuration section.
        
        Args:
            section: Section name (llm, plot, data, security, context, logging)
            
        Returns:
            Configuration section object
        """
        section_map = {
            'llm': self.llm_config,
            'plot': self.plot_config,
            'data': DataConfig(),
            'security': SecurityConfig(),
            'context': ContextConfig(),
            'logging': self.logging_config
        }
        
        if section not in section_map:
            raise ValueError(f"Unknown configuration section: {section}")
        
        return section_map[section]
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        config_obj = self.get_config_section(section)
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
            self.log_info(f"Updated config: {section}.{key} = {value}")
        else:
            raise ValueError(f"Unknown configuration key: {section}.{key}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration
        """
        return {
            'llm': self.llm_config.__dict__,
            'plot': self.plot_config.__dict__,
            'data': DataConfig().__dict__,
            'security': SecurityConfig().__dict__,
            'context': ContextConfig().__dict__,
            'logging': self.logging_config.__dict__,
            'api_keys': {k: '***' if v else None for k, v in self.api_keys.items()}
        }


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> ConfigManager:
    """Get the global configuration manager instance (alias)."""
    return get_config_manager()
