import io
import contextlib
import time
import re
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from typing import Any, Optional
from utils.config_manager import get_config_manager
from utils.code_fixer import CodeFixer
from utils.prompts import PromptManager
from utils.fuzzy_matcher import FuzzyMatcher
from utils.base_dataframe_manager import BaseDataFrameManager
from utils.logger_mixin import LoggerMixin

# Utility function for best output formatting
def format_output(result: Any, output_format: str = "auto", max_table_rows: int = 50, max_table_width: int = 20) -> str:

    if output_format == "table":
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "(Empty DataFrame)"
            # Limit rows and columns for display
            display_df = result.copy()
            if len(display_df) > max_table_rows:
                display_df = display_df.head(max_table_rows)
            if display_df.shape[1] > max_table_width:
                display_df = display_df.iloc[:, :max_table_width]
            return display_df.to_markdown(index=False)
        elif isinstance(result, pd.Series):
            return result.to_frame().to_markdown(index=True)
        else:
            return str(result)
    elif output_format == "string":
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return str(result.head(3))
        return str(result)
    elif output_format == "summary":
        if isinstance(result, pd.DataFrame):
            return f"DataFrame: shape={result.shape}, columns={list(result.columns)}"
        elif isinstance(result, pd.Series):
            return f"Series: name={result.name}, length={len(result)}"
        else:
            return str(result)
    elif output_format == "auto":
        # Best effort: DataFrame (small) as table, Series as table, scalar as string
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "(Empty DataFrame)"
            if result.shape[0] <= max_table_rows and result.shape[1] <= max_table_width:
                return result.to_markdown(index=False)
            else:
                # Summarize large DataFrame
                return f"DataFrame: shape={result.shape}, columns={list(result.columns)}\nSample:\n" + result.head(3).to_markdown(index=False)
        elif isinstance(result, pd.Series):
            if len(result) <= max_table_rows:
                return result.to_frame().to_markdown(index=True)
            else:
                return f"Series: name={result.name}, length={len(result)}\nSample:\n" + result.head(3).to_frame().to_markdown(index=True)
        elif isinstance(result, (int, float, str, bool, np.generic)):
            return str(result)
        elif result is None:
            return "(No result)"
        else:
            return str(result)
    else:
        # Fallback
        return str(result)

class QueryExecutor(BaseDataFrameManager, LoggerMixin):
    """Handles query execution, retry logic, and result processing."""
    
    def __init__(self, llm, fuzzy_matcher: FuzzyMatcher, df: pd.DataFrame, plot_manager):
        super().__init__(df)
        LoggerMixin.__init__(self)
        self.llm = llm
        self.fuzzy_matcher = fuzzy_matcher
        self.code_fixer = CodeFixer(fuzzy_matcher)
        self.prompt_manager = PromptManager()
        self.plot_manager = plot_manager
    
    def _on_dataframe_update(self, new_df: pd.DataFrame) -> None:
        """Handle DataFrame updates - update the internal DataFrame."""
        self.df = new_df.copy()
        self.logger.info(f"QueryExecutor DataFrame updated: {self.df.shape}")
    
    def execute_query_with_retry(self, query: str, output_format: str = None):
        """Execute query with retry logic and best output formatting."""
        config = get_config_manager()
        output_format = output_format or config.data_config.default_output_format
        max_retries = config.llm_config.max_retries
        retry_delay = config.llm_config.retry_delay
        max_table_rows = config.data_config.max_table_rows
        max_table_width = config.data_config.max_table_width
        
        for attempt in range(max_retries):
            try:
                self.log_query_execution(query, attempt + 1)
                # Compose prompt
                prompt = self.prompt_manager.get_query_prompt(query, self.df, output_format=output_format)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                raw_response = response.content if hasattr(response, 'content') else str(response)
                code = self._extract_valid_code(raw_response)
                if code:
                    # Fix code if needed
                    code = self.code_fixer.fix_column_names_in_code(code)
                    code = self.code_fixer.fix_pandas_syntax(code)
                    if not self.code_fixer.is_safe_code(code):
                        return "❌ Unsafe code detected. Aborting.", None
                    # Try to execute code
                    formatted_result, raw_result = self._execute_code_safely(code)
                    return formatted_result, raw_result
                else:
                    # If no code, just return the raw response, formatted
                    return format_output(raw_response, output_format, max_table_rows, max_table_width), raw_response
            except KeyError as e:
                return self._handle_key_error(e), None
            except Exception as e:
                self.log_warning(f"Error in query execution (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return f"❌ Error after {max_retries} attempts: {str(e)}", None

    def _extract_valid_code(self, raw_response: str) -> Optional[str]:
        """Extract valid code from LLM response."""
        lines = raw_response.splitlines()
        code_lines = []
        in_code_block = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for code block markers
            if line.startswith("```python") or line.startswith("```"):
                in_code_block = not in_code_block
                continue
                
            # If we're in a code block, collect the lines
            if in_code_block:
                code_lines.append(line)
            # If not in code block, look for lines that start with df or print
            elif line.startswith("df") or line.startswith("print"):
                code_lines.append(line)
        
        if not code_lines:
            return None
        
        # Join all code lines
        code = "\n".join(code_lines)
        
        # Basic validation - check for obvious syntax errors
        if self.code_fixer.has_obvious_syntax_errors(code):
            logger.warning(f"Code has obvious syntax errors, attempting to fix: {code}")
            code = self.code_fixer.attempt_syntax_fix(code)
            logger.info(f"After basic syntax fix: {code}")
        
        return code

    def _execute_code_safely(self, code: str) -> tuple[str, Any]:
        """Execute code safely in a restricted environment."""
        config = get_config_manager()
        stdout = io.StringIO()
        result = None

        try:
            if config.security_config.security_level == "production":
                # Production: Most restrictive environment
                local_vars = {"df": self.df, "pd": pd, "print": print}
                with contextlib.redirect_stdout(stdout):
                    result = eval(code, {"__builtins__": {}}, local_vars)
            else:
                # Staging/Development: More permissive environment
                self.logger.info(f"Running in {config.security_config.security_level} mode with relaxed restrictions")
                local_vars = {
                    "df": self.df, "pd": pd, "print": print, 
                    "len": len, "sum": sum, "max": max, "min": min,
                    "sorted": sorted, "list": list, "dict": dict, "set": set,
                    "create_plot": self.plot_manager.create_plot
                }
                with contextlib.redirect_stdout(stdout):
                    result = eval(code, {"__builtins__": {}}, local_vars)

            captured_output = stdout.getvalue()

            if captured_output:
                return captured_output.strip(), result
            elif result is not None:
                return self._simple_format_result(result), result
            else:
                return "Code executed successfully.", None

        except KeyError as e:
            return self._handle_key_error(e), None
        except Exception as e:
            self.logger.error(f"Code execution error: {str(e)}")
            return f"Error executing code: {str(e)}", None

    def _simple_format_result(self, result: Any) -> str:
        """Format result using the best output format utility."""
        config = get_config_manager()
        max_table_rows = config.data_config.max_table_rows
        max_table_width = config.data_config.max_table_width
        return format_output(result, "auto", max_table_rows, max_table_width)

    def _handle_key_error(self, error: KeyError) -> str:
        error_msg = str(error)
        self.log_warning(f"KeyError: {error_msg}")
        col_match = re.search(r"'([^']+)'", error_msg)
        if col_match:
            prob_col = col_match.group(1)
            suggested_col = self.fuzzy_matcher.fuzzy_match_column(prob_col)
            if suggested_col:
                return f"Column '{prob_col}' not found. Did you mean '{suggested_col}'? Available columns: {', '.join(self.df.columns)}"
        return f"Column not found: {error_msg}. Available columns: {', '.join(self.df.columns)}"
