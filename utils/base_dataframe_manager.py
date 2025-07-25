"""
Base DataFrame Manager for eliminating duplicate update_dataframe methods.
Provides a common interface for classes that need to manage DataFrame updates.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class BaseDataFrameManager(ABC):
    """
    Base class for managing DataFrame updates across different components.
    
    This class provides a common interface for updating DataFrames and ensures
    that all components that manage DataFrames follow the same pattern.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataFrame manager.
        
        Args:
            df: Initial DataFrame to manage
        """
        self.df = df
    
    def update_dataframe(self, new_df: pd.DataFrame) -> None:
        """
        Update the DataFrame and trigger any necessary side effects.
        
        Args:
            new_df: New DataFrame to set
        """
        self.df = new_df
        self._on_dataframe_update(new_df)
    
    @abstractmethod
    def _on_dataframe_update(self, new_df: pd.DataFrame) -> None:
        """
        Handle side effects when DataFrame is updated.
        
        This method should be implemented by subclasses to handle any
        additional logic needed when the DataFrame changes.
        
        Args:
            new_df: The new DataFrame that was set
        """
        # This method should be implemented by subclasses
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current DataFrame.
        
        Returns:
            Current DataFrame
        """
        return self.df
    
    def get_dataframe_shape(self) -> tuple:
        """
        Get the shape of the current DataFrame.
        
        Returns:
            DataFrame shape as (rows, columns)
        """
        return self.df.shape
    
    def get_dataframe_columns(self) -> list:
        """
        Get the columns of the current DataFrame.
        
        Returns:
            List of column names
        """
        return list(self.df.columns)
    
    def is_dataframe_empty(self) -> bool:
        """
        Check if the current DataFrame is empty.
        
        Returns:
            True if DataFrame is empty, False otherwise
        """
        return self.df.empty 