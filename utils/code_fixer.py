import re
from typing import Optional
from utils.logger import setup_logger
from utils.fuzzy_matcher import FuzzyMatcher
from utils.config_manager import get_config_manager

logger = setup_logger(__name__)

class CodeFixer:
    """Handles fixing common pandas syntax errors and column name issues."""
    
    def __init__(self, fuzzy_matcher):
        assert fuzzy_matcher is not None, "FuzzyMatcher must not be None. Please ensure you pass a valid FuzzyMatcher instance to CodeFixer."
        self.fuzzy_matcher = fuzzy_matcher
    
    def fix_column_names_in_code(self, code: str) -> str:
        """Fix column names in generated code using fuzzy matching."""
        logger.debug(f"Fixing column names in code: {code}")

        # Find all potential column references
        column_patterns = [
            r"df\[(['\"])([^'\"]+)\1\]",  # df['column'] or df["column"]
            r"df\.([a-zA-Z_][a-zA-Z0-9_]*)",  # df.column
            r"groupby\(['\"]([^'\"]+)['\"]\)",  # groupby('column')
            r"sort_values\(['\"]([^'\"]+)['\"]\)",  # sort_values('column')
        ]

        fixed_code = code

        for pattern in column_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle patterns that capture quotes
                    if len(match) == 2:
                        column_name = match[1]
                        quote_char = match[0]
                    else:
                        column_name = match[0]
                        quote_char = "'"
                else:
                    column_name = match
                    quote_char = "'"

                # Try to find matching column
                actual_column = self.fuzzy_matcher.fuzzy_match_column(column_name)
                if actual_column and actual_column != column_name:
                    # Replace in code
                    if "df." in pattern:
                        # For df.column pattern, replace with df['column']
                        fixed_code = fixed_code.replace(f"df.{column_name}", f"df['{actual_column}']")
                    else:
                        # For quoted patterns
                        fixed_code = fixed_code.replace(f"'{column_name}'", f"'{actual_column}'")
                        fixed_code = fixed_code.replace(f'"{column_name}"', f'"{actual_column}"')

        if fixed_code != code:
            logger.info(f"Fixed column names in code:\nOriginal: {code}\nFixed: {fixed_code}")

        return fixed_code

    def fix_pandas_syntax(self, code: str) -> str:
        """Fix common pandas syntax errors in generated code."""
        # Fix tuple column selection to list
        tuple_pattern = r"df\[\s*\(\s*([^)]+)\s*\)\s*\]"

        def replace_tuple_with_list(match):
            content = match.group(1)
            return f"df[[{content}]]"

        fixed_code = re.sub(tuple_pattern, replace_tuple_with_list, code)

        # Fix groupby with tuple
        groupby_tuple_pattern = r"groupby\(\s*\(\s*([^)]+)\s*\)\s*\)"
        fixed_code = re.sub(groupby_tuple_pattern, r"groupby([\1])", fixed_code)

        # Fix agg with tuple
        agg_tuple_pattern = r"agg\(\s*\(\s*([^)]+)\s*\)\s*\)"
        fixed_code = re.sub(agg_tuple_pattern, r"agg([\1])", fixed_code)

        # Fix unclosed brackets and parentheses
        fixed_code = self._fix_unclosed_brackets(fixed_code)

        if fixed_code != code:
            logger.info(f"Fixed pandas syntax:\nOriginal: {code}\nFixed: {fixed_code}")

        return fixed_code

    def _fix_unclosed_brackets(self, code: str) -> str:
        """Fix unclosed brackets, parentheses, and quotes in code."""
        # Count brackets and add missing closing ones
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        open_parens = code.count('(')
        close_parens = code.count(')')
        open_braces = code.count('{')
        close_braces = code.count('}')
        
        # Add missing closing brackets
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            code += ']' * missing_brackets
            logger.debug(f"Added {missing_brackets} missing closing brackets")
        
        # Add missing closing parentheses
        if open_parens > close_parens:
            missing_parens = open_parens - close_parens
            code += ')' * missing_parens
            logger.debug(f"Added {missing_parens} missing closing parentheses")
        
        # Add missing closing braces
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            code += '}' * missing_braces
            logger.debug(f"Added {missing_braces} missing closing braces")
        
        # Fix common incomplete patterns
        # Fix df[ without closing bracket
        if code.strip().endswith('df['):
            code += ']'
            logger.debug("Fixed incomplete df[ pattern")
        
        # Fix df[['column' without closing brackets
        if code.strip().endswith("df[['") or code.strip().endswith("df[[\""):
            code += "']]"
            logger.debug("Fixed incomplete df[['column']] pattern")
        
        # Fix print( without closing parenthesis
        if code.strip().endswith('print('):
            code += ')'
            logger.debug("Fixed incomplete print( pattern")
        
        return code

    def has_obvious_syntax_errors(self, code: str) -> bool:
        """Check for obvious syntax errors in the code."""
        # Check for unclosed brackets
        if code.count('[') != code.count(']'):
            return True
        if code.count('(') != code.count(')'):
            return True
        if code.count('{') != code.count('}'):
            return True
        # Check for incomplete patterns
        if code.strip().endswith('df['):
            return True
        if code.strip().endswith('print('):
            return True
        return False

    def attempt_syntax_fix(self, code: str) -> str:
        """Attempt to fix obvious syntax errors in the code."""
        # This is a basic fix - the main fixing happens in fix_pandas_syntax
        # Just ensure we have at least one complete statement
        if code.strip().endswith('df['):
            code += ']'
        if code.strip().endswith('print('):
            code += ')'
        return code

    def is_safe_code(self, code: str) -> bool:
        """Check if the code is safe to execute. Only block high-risk patterns in non-production."""
        code_lower = code.lower()
        config = get_config_manager()
        # Only check code outside of string literals
        def remove_string_literals(s):
            return re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", "", s)
        code_no_strings = remove_string_literals(code_lower)
        # Always check high-risk patterns (now stricter)
        for forbidden in config.security_config.forbidden_code_patterns:
            # Only match forbidden as function call or statement, not substring
            if re.search(rf"\\b{re.escape(forbidden)}", code_no_strings):
                logger.warning(f"High-risk pattern detected and blocked: {forbidden}")
                return False
        # Only block medium-risk patterns in production
        if config.security_config.security_level == "production":
            for pattern in config.security_config.staging_allowed_patterns:
                if re.search(rf"\\b{re.escape(pattern)}", code_no_strings):
                    logger.warning(f"Medium-risk pattern blocked in production: {pattern}")
                    return False
        else:
            # In staging/development, allow medium-risk patterns but log them
            for pattern in config.security_config.staging_allowed_patterns:
                if re.search(rf"\\b{re.escape(pattern)}", code_no_strings):
                    logger.info(f"Medium-risk pattern allowed in {config.security_config.security_level}: {pattern}")
        return True 