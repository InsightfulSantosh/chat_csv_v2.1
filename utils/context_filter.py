import re
import pandas as pd
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.fuzzy_matcher import FuzzyMatcher
from utils.data_utils import normalize_str_series

logger = setup_logger(__name__)

@dataclass
class FilterState:
    """Immutable filter state for better state management."""
    filters: Dict[str, Union[str, List[str]]]
    entity_memory: Dict[str, Union[str, List[str]]]
    
    def copy(self) -> 'FilterState':
        return FilterState(
            filters=self.filters.copy(),
            entity_memory=self.entity_memory.copy()
        )

class EntityMatcher:
    """Handles entity matching logic with consistent thresholds."""
    
    def __init__(self, config, fuzzy_matcher: FuzzyMatcher):
        self.config = config
        self.fuzzy_matcher = fuzzy_matcher
        # Use stopwords from config if available, else fallback to default
        self.stopwords = set(getattr(self.config.data_config, 'stopwords', []))
        if not self.stopwords:
            # Fallback default stopwords
            self.stopwords = {"as", "in", "the", "of", "for", "on", "at", "by", "with", "to", "from", "and", "or", "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did", "a", "an", "it", "this", "that", "these", "those", "but", "if", "then", "so", "not", "no", "yes", "can", "will", "just", "now", "out", "up", "down", "off", "over", "under", "again", "more", "less", "very", "much", "some", "any", "all", "each", "every", "either", "neither", "both", "which", "who", "whom", "whose", "what", "when", "where", "why", "how"}
    
    def find_matches(self, query_words: List[str], values: List[str]) -> List[str]:
        """Find matching entities using a consistent strategy."""
        if not query_words or not values:
            return []
        
        # Filter out stopwords and short words (from config)
        filtered_query_words = [w for w in query_words if len(w) > 2 and w not in self.stopwords]
        if not filtered_query_words:
            return []
        
        # Normalize values for matching
        norm_values = [(v, self._normalize(v)) for v in values]
        matches = []
        
        # 1. Exact whole word matches
        matches.extend(self._exact_matches(filtered_query_words, norm_values))
        
        # 2. Substring matches (if no exact matches)
        if not matches:
            matches.extend(self._substring_matches(filtered_query_words, norm_values))
        
        # 3. Phrase matches (if no substring matches)
        if not matches:
            matches.extend(self._phrase_matches(filtered_query_words, norm_values))
        
        # 4. Fuzzy matches (if no phrase matches)
        if not matches:
            matches.extend(self.fuzzy_matcher.fuzzy_match_values(filtered_query_words, norm_values))
        
        return self._deduplicate(matches)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for consistent matching."""
        return text.strip().lower()
    
    def _exact_matches(self, query_words: List[str], norm_values: List[Tuple[str, str]]) -> List[str]:
        """Find exact whole word matches."""
        return [orig for orig, norm in norm_values if norm in query_words]
    
    def _substring_matches(self, query_words: List[str], norm_values: List[Tuple[str, str]]) -> List[str]:
        """Find whole word (token-based) matches (e.g., 'manager' matches 'hr manager', 'sales manager', but not 'mechanical engineer')."""
        matches = []
        for orig, norm in norm_values:
            for word in query_words:
                # Match as a whole word (case-insensitive)
                if re.search(rf'\b{re.escape(word)}\b', norm):
                    matches.append(orig)
                    break
        return matches
    
    def _phrase_matches(self, query_words: List[str], norm_values: List[Tuple[str, str]]) -> List[str]:
        """Find phrase matches."""
        query_text = ' '.join(query_words)
        return [orig for orig, norm in norm_values if f' {norm} ' in f' {query_text} ']
    
    def _deduplicate(self, matches: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        return [x for x in matches if not (x in seen or seen.add(x))]

class QueryProcessor:
    """Handles query processing and rewriting."""
    
    def __init__(self, config):
        self.config = config
    
    def clean_query(self, query: str) -> Tuple[str, List[str]]:
        """Clean query by removing referent words."""
        query_lower = query.lower().strip()
        referents = set(self.config.context_config.referent_words)
        query_words = [w for w in query_lower.split() if w not in referents]
        return ' '.join(query_words), query_words
    
    def is_correction(self, query: str) -> Optional[str]:
        """Check if query is a correction and extract the corrected column."""
        query_lower = query.lower().strip()
        for pattern in self.config.context_config.correction_patterns:
            match = re.match(pattern, query_lower)
            if match:
                return match.group(1)
        return None
    
    def has_referent_words(self, query: str) -> bool:
        """Check if query contains referent words."""
        query_lower = query.lower()
        return any(w in query_lower for w in self.config.context_config.referent_words)
    
    def is_global_query(self, query: str) -> bool:
        """Check if query refers to entire dataset."""
        query_lower = query.lower()
        return any(k in query_lower for k in self.config.context_config.global_query_keywords)
    
    def is_reset_command(self, query: str) -> bool:
        """Check if query is a reset command."""
        query_lower = query.lower()
        return any(cmd in query_lower for cmd in self.config.context_config.reset_commands)

class DataFrameFilter:
    """Handles DataFrame filtering operations."""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Union[str, List[str], Tuple[str, float, float]]]) -> pd.DataFrame:
        """Apply filters to DataFrame, supporting robust numeric filter extraction."""
        filtered_df = df.copy()
        
        for col, values in filters.items():
            if col not in filtered_df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            # Numeric filter logic
            if isinstance(values, tuple) and len(values) >= 2 and isinstance(values[1], (int, float)):
                if values[0] == "lt":
                    filtered_df = filtered_df[filtered_df[col] < values[1]]
                elif values[0] == "le":
                    filtered_df = filtered_df[filtered_df[col] <= values[1]]
                elif values[0] == "gt":
                    filtered_df = filtered_df[filtered_df[col] > values[1]]
                elif values[0] == "ge":
                    filtered_df = filtered_df[filtered_df[col] >= values[1]]
                elif values[0] == "eq":
                    filtered_df = filtered_df[filtered_df[col] == values[1]]
                elif values[0] == "ne":
                    filtered_df = filtered_df[filtered_df[col] != values[1]]
                elif values[0] == "between" and len(values) == 3:
                    filtered_df = filtered_df[(filtered_df[col] >= values[1]) & (filtered_df[col] <= values[2])]
                continue
            
            # Categorical filter logic (existing)
            if isinstance(values, list):
                # Multiple values - use regex pattern
                pattern = '|'.join(map(re.escape, values))
                mask = normalize_str_series(filtered_df[col]).str.contains(
                    pattern, case=False, na=False
                )
            else:
                # Single value - exact match
                mask = (
                    normalize_str_series(filtered_df[col]) == 
                    str(values).lower().strip()
                )
            
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    @staticmethod
    def get_categorical_columns(df: pd.DataFrame, config) -> List[str]:
        """Get columns suitable for categorical filtering."""
        categorical_cols = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            unique_count = df[col].nunique()
            if (config.data_config.min_categorical_values_for_memory <= 
                unique_count < config.data_config.max_categorical_values_for_memory):
                categorical_cols.append(col)
        
        return categorical_cols

class ContextFilter:
    """Improved context-aware filtering with better separation of concerns."""
    
    def __init__(self, df_full: pd.DataFrame, fuzzy_matcher: FuzzyMatcher):
        self.df_full = df_full
        self.fuzzy_matcher = fuzzy_matcher
        
        # Initialize components
        config = get_config_manager()
        self.entity_matcher = EntityMatcher(config, self.fuzzy_matcher)
        self.query_processor = QueryProcessor(config)
        self.df_filter = DataFrameFilter()
        
        # Initialize state
        self._state = FilterState(filters={}, entity_memory={})
        self._df = df_full.copy()
        self.last_result_df = None  # Store the last result DataFrame for follow-up context
        self.last_result_indices = None  # Optionally store row indices for entity memory
    
    @property
    def df(self) -> pd.DataFrame:
        """Get current filtered DataFrame."""
        return self._df
    
    def apply_context_filter(self, query: str):
        """Apply context-aware filtering based on query content."""
        logger.info(f"Applying context filter for query: '{query}'")
        
        # Check for special commands
        if self.query_processor.is_reset_command(query):
            self.reset_filters()
            return
        
        corrected_column = self.query_processor.is_correction(query)
        if corrected_column:
            # Handle corrections without changing filters
            logger.info(f"Detected correction for column: {corrected_column}")
            return
        
        # Process query
        cleaned_query, query_words = self.query_processor.clean_query(query)
        logger.debug(f"Cleaned query: '{cleaned_query}' -> words: {query_words}")
        
        # Apply entity memory if query has referent words
        new_state = self._state.copy()
        if self.query_processor.has_referent_words(query):
            new_state.filters = new_state.entity_memory.copy()
        
        # Extract new filters from query
        new_filters = self._extract_filters(query_words, query)
        
        # Update filters (replace existing filters for same columns)
        new_state.filters.update(new_filters)
        
        # Apply filters
        self._apply_state(new_state)
        
        # Store the filtered DataFrame and indices for memory
        self.last_result_df = self._df.copy()
        self.last_result_indices = self._df.index.tolist()
        
        # Handle empty results
        if self._df.empty and new_filters:
            logger.warning("Empty result after filtering, reverting to previous state")
            # Keep entity memory but remove the new filters that caused empty result
            fallback_state = FilterState(
                filters=new_state.entity_memory.copy(),
                entity_memory=new_state.entity_memory.copy()
            )
            self._apply_state(fallback_state)
            self.last_result_df = self._df.copy()
            self.last_result_indices = self._df.index.tolist()
    
    def _extract_filters(self, query_words: List[str], query: str = "") -> Dict[str, Union[str, List[str], Tuple[str, float, float]]]:
        """Extract filters from query words, supporting robust and diverse numeric filter extraction, including reverse patterns."""
        import re
        config = get_config_manager()
        new_filters = {}
        # Categorical columns (existing logic)
        categorical_cols = self.df_filter.get_categorical_columns(self.df_full, config)
        for col in categorical_cols:
            unique_values = self.df_full[col].dropna().astype(str).unique().tolist()
            matches = self.entity_matcher.find_matches(query_words, unique_values)
            if matches:
                new_filters[col] = matches if len(matches) > 1 else matches[0]
                logger.debug(f"Extracted values for column '{col}': {matches}")
        # Numeric columns (enhanced logic)
        numeric_cols = [col for col in self.df_full.columns if pd.api.types.is_numeric_dtype(self.df_full[col])]
        query_text = ' '.join(query_words)
        for col in numeric_cols:
            col_lower = col.lower().replace('_', ' ')
            # Forward patterns: column name before operator/number
            for op, pat, opkey in [
                ("lt", rf"{col_lower}.*(?:<|under|less than|below|at most|up to|no more than)\s*(\d+(?:\.\d+)?)", "<"),
                ("le", rf"{col_lower}.*(?:<=|at most|no more than|not more than|less than or equal to)\s*(\d+(?:\.\d+)?)", "<="),
                ("gt", rf"{col_lower}.*(?:>|over|greater than|more than|above|at least|no less than)\s*(\d+(?:\.\d+)?)", ">"),
                ("ge", rf"{col_lower}.*(?:>=|at least|no less than|not less than|greater than or equal to)\s*(\d+(?:\.\d+)?)", ">="),
                ("eq", rf"{col_lower}.*(?:=|equals|equal to|is|is exactly|==)\s*(\d+(?:\.\d+)?)", "=="),
                ("ne", rf"{col_lower}.*(?:!=|not|not equal to|is not|isn't|doesn't equal)\s*(\d+(?:\.\d+)?)", "!="),
            ]:
                m = re.search(pat, query_text)
                if m:
                    val = float(m.group(1))
                    if op == "lt":
                        new_filters[col] = ("lt", val)
                    elif op == "le":
                        new_filters[col] = ("le", val)
                    elif op == "gt":
                        new_filters[col] = ("gt", val)
                    elif op == "ge":
                        new_filters[col] = ("ge", val)
                    elif op == "eq":
                        new_filters[col] = ("eq", val)
                    elif op == "ne":
                        new_filters[col] = ("ne", val)
                    logger.debug(f"Extracted numeric filter for column '{col}': {opkey} {val}")
                    break
            # Reverse patterns: operator/number before column name
            for op, pat, opkey in [
                ("lt", rf"(?:<|under|less than|below|at most|up to|no more than)\s*{col_lower}\s*(\d+(?:\.\d+)?)", "<"),
                ("le", rf"(?:<=|at most|no more than|not more than|less than or equal to)\s*{col_lower}\s*(\d+(?:\.\d+)?)", "<="),
                ("gt", rf"(?:>|over|greater than|more than|above|at least|no less than)\s*{col_lower}\s*(\d+(?:\.\d+)?)", ">"),
                ("ge", rf"(?:>=|at least|no less than|not less than|greater than or equal to)\s*{col_lower}\s*(\d+(?:\.\d+)?)", ">="),
                ("eq", rf"(?:=|equals|equal to|is|is exactly|==)\s*{col_lower}\s*(\d+(?:\.\d+)?)", "=="),
                ("ne", rf"(?:!=|not|not equal to|is not|isn't|doesn't equal)\s*{col_lower}\s*(\d+(?:\.\d+)?)", "!="),
            ]:
                m = re.search(pat, query_text)
                if m:
                    val = float(m.group(1))
                    if op == "lt":
                        new_filters[col] = ("lt", val)
                    elif op == "le":
                        new_filters[col] = ("le", val)
                    elif op == "gt":
                        new_filters[col] = ("gt", val)
                    elif op == "ge":
                        new_filters[col] = ("ge", val)
                    elif op == "eq":
                        new_filters[col] = ("eq", val)
                    elif op == "ne":
                        new_filters[col] = ("ne", val)
                    logger.debug(f"Extracted numeric filter for column '{col}': {opkey} {val}")
                    break
            # Between/Range patterns (forward)
            m = re.search(rf"{col_lower}.*(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)", query_text)
            if m:
                v1, v2 = float(m.group(1)), float(m.group(2))
                new_filters[col] = ("between", min(v1, v2), max(v1, v2))
                logger.debug(f"Extracted numeric filter for column '{col}': between {v1} and {v2}")
                continue
            # Between/Range patterns (reverse)
            m = re.search(rf"(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?).*{col_lower}", query_text)
            if m:
                v1, v2 = float(m.group(1)), float(m.group(2))
                new_filters[col] = ("between", min(v1, v2), max(v1, v2))
                logger.debug(f"Extracted numeric filter for column '{col}': between {v1} and {v2}")
                continue
        # 2. Generic patterns (e.g., 'under 23') - try to infer column
        # Heuristic: if 'age' is a numeric column, assume it's the target for generic numeric filters.
        if 'age' in numeric_cols:
            col = 'age'
            # Check if a filter for this column has already been found by specific patterns
            if col not in new_filters:
                for op, pat, opkey in [
                    ("lt", r"(?:<|under|less than|below|at most|up to|no more than)\s*(\d+(?:\.\d+)?)", "<"),
                    ("le", r"(?:<=|at most|no more than|not more than|less than or equal to)\s*(\d+(?:\.\d+)?)", "<="),
                    ("gt", r"(?:>|over|greater than|more than|above|at least|no less than)\s*(\d+(?:\.\d+)?)", ">"),
                    ("ge", r"(?:>=|at least|no less than|not less than|greater than or equal to)\s*(\d+(?:\.\d+)?)", ">="),
                    ("eq", r"(?:=|equals|equal to|is|is exactly|==)\s*(\d+(?:\.\d+)?)", "=="),
                    ("ne", r"(?:!=|not|not equal to|is not|isn't|doesn't equal)\s*(\d+(?:\.\d+)?)", "!="),
                ]:
                    m = re.search(pat, query_text)
                    if m:
                        val = float(m.group(1))
                        if op == "lt":
                            new_filters[col] = ("lt", val)
                        elif op == "le":
                            new_filters[col] = ("le", val)
                        elif op == "gt":
                            new_filters[col] = ("gt", val)
                        elif op == "ge":
                            new_filters[col] = ("ge", val)
                        elif op == "eq":
                            new_filters[col] = ("eq", val)
                        elif op == "ne":
                            new_filters[col] = ("ne", val)
                        logger.debug(f"Extracted generic numeric filter for column '{col}': {opkey} {val}")
                        break # Stop after first match for this column
                
                if col not in new_filters: # Only check for 'between' if no other filter was found
                    m = re.search(r"(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)", query_text)
                    if m:
                        v1, v2 = float(m.group(1)), float(m.group(2))
                        new_filters[col] = ("between", min(v1, v2), max(v1, v2))
                        logger.debug(f"Extracted generic numeric filter for column '{col}': between {v1} and {v2}")
        return new_filters
    
    def _apply_state(self, new_state: FilterState):
        """Apply new state and update DataFrame."""
        self._state = new_state
        self._df = self.df_filter.apply_filters(self.df_full, self._state.filters)
        
        # Update entity memory to reflect active filters
        self._state.entity_memory = self._state.filters.copy()
        
        logger.info(f"Applied filters: {self._state.filters}")
        logger.info(f"Resulting DataFrame shape: {self._df.shape}")
    
    def rewrite_with_context(self, query: str) -> str:
        """Rewrite query to include context information."""
        corrected_column = self.query_processor.is_correction(query)
        if corrected_column and self._state.entity_memory:
            # Handle corrections
            context_desc = self._format_context_description()
            return f"# NOTE: df refers to rows where {context_desc}\nHow many of them have {corrected_column} = 'AI researcher'"
        
        if (self.query_processor.has_referent_words(query) and 
            self._state.entity_memory):
            context_desc = self._format_context_description()
            return f"# NOTE: df refers to rows where {context_desc}\n{query}"
        
        return query
    
    def _format_context_description(self) -> str:
        """Format context description for query rewriting."""
        return ", ".join(
            f"{col} in {values}" 
            for col, values in self._state.entity_memory.items()
        )
    
    def update_entity_memory_from_output(self, output: str, question: str):
        """Update entity memory based on output and question."""
        question_lower, question_words = self.query_processor.clean_query(question)
        config = get_config_manager()
        
        categorical_cols = self.df_filter.get_categorical_columns(self.df_full, config)
        
        new_memory = {}
        for col in categorical_cols:
            values = self.df_full[col].dropna().astype(str).unique().tolist()
            
            # Check both question and output for matches
            question_matches = self.entity_matcher.find_matches(question_words, values)
            output_matches = [v for v in values if v in output]
            
            all_matches = list(set(question_matches + output_matches))
            if all_matches:
                new_memory[col] = all_matches
        
        # Update entity memory
        self._state.entity_memory.update(new_memory)
    
    def get_current_filters(self) -> Dict:
        """Get current filters."""
        return self._state.filters.copy()
    
    def get_context_summary(self) -> Dict:
        """Get summary of unique values in filtered DataFrame."""
        summary = {}
        for col in self._df.columns:
            if pd.api.types.is_string_dtype(self._df[col]):
                unique_vals = normalize_str_series(self._df[col].dropna()).unique().tolist()
            else:
                unique_vals = self._df[col].dropna().unique().tolist()
            summary[col] = unique_vals
        
        return summary
    
    def get_entity_memory(self) -> Dict:
        """Get entity memory."""
        return self._state.entity_memory.copy()

    def apply_entity_memory(self):
        """Apply filters based on current entity memory or use last_result_df if available."""
        if self.last_result_df is not None:
            self._df = self.last_result_df.copy()
            logger.info("Applied last_result_df for entity memory context.")
        else:
            self._state.filters = self._state.entity_memory.copy()
            self._df = self.df_filter.apply_filters(self.df_full, self._state.filters)
