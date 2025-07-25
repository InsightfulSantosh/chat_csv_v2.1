"""
Logger Mixin for standardized logging across all classes.
Provides consistent logging behavior and reduces code duplication.
"""

import logging
from typing import Optional
from utils.logger import setup_logger


class LoggerMixin:
    """
    Mixin class that provides standardized logging functionality.
    
    This mixin can be used by any class that needs logging capabilities.
    It automatically sets up a logger with the class name and provides
    convenient logging methods.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the logger mixin."""
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger instance for this class.
        
        Returns:
            Logger instance configured for this class
        """
        if self._logger is None:
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger
    
    def log_info(self, message: str, *args, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.info(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.debug(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.error(message, *args, **kwargs)
    
    def log_critical(self, message: str, *args, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.critical(message, *args, **kwargs)
    
    def log_exception(self, message: str, exc_info: bool = True, *args, **kwargs) -> None:
        """
        Log an exception with traceback.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception info
            *args: Additional arguments for string formatting
            **kwargs: Additional keyword arguments for string formatting
        """
        self.logger.exception(message, exc_info=exc_info, *args, **kwargs)
    
    def log_method_entry(self, method_name: Optional[str] = None) -> None:
        """
        Log method entry for debugging.
        
        Args:
            method_name: Name of the method (defaults to caller method)
        """
        if method_name is None:
            import inspect
            method_name = inspect.currentframe().f_back.f_code.co_name
        self.log_debug(f"Entering method: {method_name}")
    
    def log_method_exit(self, method_name: Optional[str] = None) -> None:
        """
        Log method exit for debugging.
        
        Args:
            method_name: Name of the method (defaults to caller method)
        """
        if method_name is None:
            import inspect
            method_name = inspect.currentframe().f_back.f_code.co_name
        self.log_debug(f"Exiting method: {method_name}")
    
    def log_performance(self, operation: str, duration: float) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        self.log_info(f"Performance - {operation}: {duration:.3f}s")
    
    def log_dataframe_info(self, df_name: str, shape: tuple, columns: list) -> None:
        """
        Log DataFrame information.
        
        Args:
            df_name: Name of the DataFrame
            shape: DataFrame shape (rows, columns)
            columns: List of column names
        """
        self.log_info(f"DataFrame '{df_name}' - Shape: {shape}, Columns: {len(columns)}")
        self.log_debug(f"DataFrame '{df_name}' columns: {columns}")
    
    def log_configuration(self, config_dict: dict) -> None:
        """
        Log configuration information.
        
        Args:
            config_dict: Configuration dictionary to log
        """
        self.log_info("Configuration loaded:")
        for key, value in config_dict.items():
            # Mask sensitive values
            if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                masked_value = '*' * min(len(str(value)), 8) if value else None
                self.log_info(f"  {key}: {masked_value}")
            else:
                self.log_info(f"  {key}: {value}")
    
    def log_llm_creation(self, provider: str, model: str, temperature: float) -> None:
        """
        Log LLM creation details.
        
        Args:
            provider: LLM provider name
            model: Model name
            temperature: Temperature setting
        """
        self.log_info(f"🤖 Creating LLM - Provider: {provider.upper()}, Model: {model}, Temperature: {temperature}")
    
    def log_plot_creation(self, plot_type: str, x_column: str = None, y_column: str = None) -> None:
        """
        Log plot creation details.
        
        Args:
            plot_type: Type of plot
            x_column: X-axis column
            y_column: Y-axis column
        """
        columns_info = f"x={x_column}" if x_column else ""
        if y_column:
            columns_info += f", y={y_column}" if columns_info else f"y={y_column}"
        self.log_info(f"📊 Creating {plot_type} plot - {columns_info}")
    
    def log_query_execution(self, query: str, attempt: int = 1) -> None:
        """
        Log query execution details.
        
        Args:
            query: Query being executed
            attempt: Attempt number (for retries)
        """
        attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
        self.log_debug(f"Executing query{attempt_info}: {query}")
    
    def log_success(self, operation: str, details: str = "") -> None:
        """
        Log successful operation.
        
        Args:
            operation: Name of the operation
            details: Additional details
        """
        message = f"✅ {operation} completed successfully"
        if details:
            message += f" - {details}"
        self.log_info(message)
    
    def log_failure(self, operation: str, error: str) -> None:
        """
        Log failed operation.
        
        Args:
            operation: Name of the operation
            error: Error message
        """
        self.log_error(f"❌ {operation} failed: {error}")
    
    def log_warning_with_suggestion(self, warning: str, suggestion: str) -> None:
        """
        Log warning with suggestion.
        
        Args:
            warning: Warning message
            suggestion: Suggestion for resolution
        """
        self.log_warning(f"⚠️ {warning} - Suggestion: {suggestion}") 