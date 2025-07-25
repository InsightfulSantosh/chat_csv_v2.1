"""
Data loader module for SmartPandasAgent.
Handles CSV loading, preprocessing, and formatting.
"""

import pandas as pd
import os
import re
from typing import Optional, Dict, Any
from utils.config_manager import get_config_manager
from utils.logger import setup_logger



def load_csv_with_preprocessing(
    input_path: str,
    lowercase: bool = True,
    rename_columns: Optional[Dict[str, str]] = None,
    save_formatted: bool = True,
    output_folder: str = "data/formated_data"
) -> pd.DataFrame:
    """
    Load and preprocess CSV with configurable options.
    
    Args:
        input_path: Path to the input CSV file
        lowercase: Whether to convert to lowercase
        rename_columns: Dictionary of column renames
        save_formatted: Whether to save the formatted version
        output_folder: Output folder for formatted files
        
    Returns:
        Preprocessed DataFrame
    """
    # Load the CSV
    df = pd.read_csv(input_path)

    # Clean column names: lowercase, replace symbols with space, collapse spaces, strip, UTF-8 encode/decode
    def clean_column_name(col):
        # Lowercase
        col = col.lower()
        # Replace non-alphanumeric with space
        col = re.sub(r'[^a-z0-9]', ' ', col)
        # Collapse multiple spaces
        col = re.sub(r'\s+', ' ', col)
        # Strip
        col = col.strip()
        # Ensure UTF-8 encoding/decoding
        col = col.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        return col
    df.columns = [clean_column_name(col) for col in df.columns]

    # Clean string data values: lowercase, ensure UTF-8 encoding/decoding
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'))
    
    # Apply custom column renames
    if rename_columns:
        df = df.rename(columns=rename_columns)
    
    # Validate and fix data types
    validate_dataframe_dtypes(df)
    
    if save_formatted:
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Define the output file name
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_folder, file_name)
        
        # Save the modified DataFrame
        df.to_csv(output_path, index=False)
    
    logger = setup_logger(__name__)
    logger.info(f"📊 DataFrame loaded: {df.shape}")
    logger.info(f"📋 Columns: {list(df.columns)}")
    
    return df


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with DataFrame information
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "null_counts": df.isnull().sum().to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns}
    }
    
    return info

def validate_dataframe_dtypes(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame data types and fix common issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = setup_logger(__name__)
    logger.info("🔍 Validating DataFrame data types...")
    
    issues_found = False
    
    for col in df.columns:
        col_dtype = df[col].dtype
        logger.debug(f"   Column '{col}': {col_dtype}")
        
        # Check for mixed data types in object columns
        if col_dtype == 'object':
            # Check if there are any boolean values mixed with strings
            bool_mask = df[col].apply(lambda x: isinstance(x, bool))
            if bool_mask.any():
                logger.warning(f"   ⚠️  Warning: Column '{col}' contains boolean values mixed with strings")
                issues_found = True
                
                # Convert boolean values to strings
                df[col] = df[col].astype(str)
                logger.info(f"   ✅ Fixed: Converted boolean values to strings in column '{col}'")
        
        # Check for numeric columns that might have string values
        elif pd.api.types.is_numeric_dtype(col_dtype):
            # Check if there are any non-numeric values
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna()
            if non_numeric.any():
                logger.warning(f"   ⚠️  Warning: Column '{col}' contains non-numeric values")
                issues_found = True
    
    if not issues_found:
        logger.info("   ✅ All data types are valid")
    
    return not issues_found


def validate_csv_path(file_path: str) -> bool:
    """
    Validate if a CSV file path exists and is accessible.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if file exists and is accessible
    """
    logger = setup_logger(__name__)
    
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    if not file_path.lower().endswith('.csv'):
        logger.error(f"❌ File is not a CSV: {file_path}")
        return False
    
    try:
        # Try to read a small sample to validate
        pd.read_csv(file_path, nrows=5)
        return True
    except Exception as e:
        logger.error(f"❌ Error reading CSV: {e}")
        return False


# Generic data processing function
def process_csv_data(input_path: str, column_renames: dict = None) -> pd.DataFrame:
    """
    Process any CSV data with configurable formatting.
    
    Args:
        input_path: Path to the CSV file
        column_renames: Dictionary of column renames (optional)
        
    Returns:
        Processed DataFrame
    """
    # Validate the input file
    if not validate_csv_path(input_path):
        raise FileNotFoundError(f"Invalid or inaccessible file: {input_path}")
    
    # Load and preprocess the data
    df = load_csv_with_preprocessing(
        input_path=input_path,
        lowercase=True,
        rename_columns=column_renames,
        save_formatted=True,
        output_folder="data/formated_data"
    )
    
    return df

# Legacy function for backward compatibility
def process_professionals_data(input_path: str = "/content/professionals_in_pg.csv") -> pd.DataFrame:
    """
    Process the professionals data with specific formatting.
    
    Args:
        input_path: Path to the professionals CSV file
        
    Returns:
        Processed DataFrame
    """
    # Define column renames for this specific dataset
    column_renames = {
        "rent (inr)": "rent"
    }
    
    return process_csv_data(input_path, column_renames)





# Example usage (commented out to avoid execution on import)
if __name__ == "__main__":
    # Example usage
    logger = setup_logger(__name__)
    try:
        df = process_professionals_data("/content/professionals_in_pg.csv")
        logger.info("✅ Data processing completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error processing data: {e}") 