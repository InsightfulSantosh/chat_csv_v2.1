import re
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
from utils.config_manager import get_config_manager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FuzzyMatcher:
    """Handles fuzzy matching for column names, values, and suggestions."""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.column_mappings = {}
        if self.df is not None:
            self._create_lookup_tables()
    
    def _create_lookup_tables(self):
        """Create lookup tables for fuzzy matching of columns."""
        if self.df is None:
            return
        logger.info("Creating lookup tables for fuzzy matching (columns only).")
        for col in self.df.columns:
            # Original column name
            self.column_mappings[col.lower().strip()] = col
            # Remove special characters and spaces
            clean_col = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
            if clean_col:
                self.column_mappings[clean_col] = col
            # Common variations
            self.column_mappings[col.lower().replace('_', '').replace(' ', '')] = col
            self.column_mappings[col.lower().replace('-', '').replace(' ', '')] = col

    def fuzzy_match_column(self, query_col: str, threshold: float = None) -> Optional[str]:
        """Find best matching column name using fuzzy matching."""
        if not self.column_mappings:
            logger.warning("FuzzyMatcher not initialized with a DataFrame. Cannot match columns.")
            return None
            
        if threshold is None:
            config = get_config_manager()
            threshold = config.data_config.fuzzy_match_threshold
            
        query_col_clean = query_col.lower().strip()

        # Exact match first
        if query_col_clean in self.column_mappings:
            return self.column_mappings[query_col_clean]

        # Fuzzy match
        best_match = None
        best_score = 0

        for mapped_col, original_col in self.column_mappings.items():
            score = SequenceMatcher(None, query_col_clean, mapped_col).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = original_col

        if best_match:
            logger.info(f"Fuzzy matched column '{query_col}' to '{best_match}' (score: {best_score:.2f})")

        return best_match

    def suggest_columns(self, query: str) -> list:
        """Suggest column names based on query."""
        if self.df is None:
            logger.warning("FuzzyMatcher not initialized with a DataFrame. Cannot suggest columns.")
            return []
            
        suggestions = []
        query_lower = query.lower()
        config = get_config_manager()

        for col in self.df.columns:
            if col.lower() in query_lower:
                suggestions.append(col)
            elif SequenceMatcher(None, col.lower(), query_lower).ratio() > config.data_config.fuzzy_match_threshold:
                suggestions.append(col)

        return suggestions

    def _get_threshold(self, text: str) -> float:
        """Get adaptive threshold for fuzzy matching."""
        config = get_config_manager()
        base_threshold = getattr(config.data_config, 'fuzzy_match_threshold', 0.75)
        # Increase threshold for short strings to avoid false positives
        if len(text) <= 4:
            return max(0.95, base_threshold)
        return base_threshold

    def fuzzy_match_values(self, query_words: List[str], norm_values: List[Tuple[str, str]]) -> List[str]:
        """
        Find fuzzy matches for values with adaptive thresholds.
        
        Args:
            query_words: A list of words from the user query.
            norm_values: A list of tuples, where each tuple contains the original value and its normalized version.
            
        Returns:
            A list of matching original values.
        """
        matches = []
        for orig, norm in norm_values:
            threshold = self._get_threshold(norm)
            # Find the best match score for the normalized value against all query words
            max_score = max(
                (SequenceMatcher(None, norm, word).ratio() for word in query_words),
                default=0
            )
            if max_score > threshold:
                matches.append(orig)
        return matches
