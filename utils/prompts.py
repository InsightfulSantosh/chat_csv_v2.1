from typing import Dict, Any, Optional, List
from utils.config_manager import get_config_manager
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)

class PromptTemplate:
    """Base class for prompt templates with variable substitution."""
    
    def __init__(self, template: str, required_vars: List[str] = None):
        self.template = template
        self.required_vars = required_vars or []
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        missing_vars = [var for var in self.required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        return self.template.format(**kwargs)

class PromptManager:
    """Enhanced prompt manager with configuration-driven templates and better structure."""
    
    def __init__(self):
        self.config = get_config_manager()
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all prompt templates."""
        self.templates = {
            'system': PromptTemplate(
                template=self._get_system_template(),
                required_vars=['schema', 'examples_text', 'security_level', 'security_rules', 'langgraph_rules', 'output_format_rules', 'code_style_rules', 'error_handling_rules']
            ),
            'query': PromptTemplate(
                template=self._get_query_template(),
                required_vars=['query', 'columns', 'dtypes', 'shape', 'sample_rows', 'examples_text', 'output_format', 'security_level', 'data_source_rules', 'column_enforcement_rules', 'syntax_compliance_rules', 'string_filtering_rules', 'ranking_queries_rules', 'output_policy_rules', 'plotting_restrictions', 'strict_prohibitions']
            ),
            'plot': PromptTemplate(
                template=self._get_plot_template(),
                required_vars=['plot_type', 'x_column', 'y_column', 'title', 'color_column', 'plot_guidelines']
            ),
            'error_correction': PromptTemplate(
                template=self._get_error_correction_template(),
                required_vars=['original_query', 'error_message', 'correction_guidelines']
            ),
            'context_aware': PromptTemplate(
                template=self._get_context_aware_template(),
                required_vars=['query', 'context_description', 'context_guidelines']
            ),
            'column_suggestion': PromptTemplate(
                template=self._get_column_suggestion_template(),
                required_vars=['query', 'available_columns', 'column_guidelines']
            ),
            'aggregation': PromptTemplate(
                template=self._get_aggregation_template(),
                required_vars=['operation', 'column', 'aggregation_guidelines']
            ),
            'filtering': PromptTemplate(
                template=self._get_filtering_template(),
                required_vars=['column', 'value', 'operation', 'filtering_guidelines']
            ),
            'data_exploration': PromptTemplate(
                template=self._get_data_exploration_template(),
                required_vars=['exploration_guidelines']
            ),
            'complex_query': PromptTemplate(
                template=self._get_complex_query_template(),
                required_vars=['query_parts', 'combine_operation', 'complex_query_guidelines']
            ),
            'query_analysis': PromptTemplate(
                template=self._get_query_analysis_template(),
                required_vars=['query', 'columns']
            ),
            'complex_query_parser': PromptTemplate(
                template=self._get_complex_query_parser_template(),
                required_vars=['query']
            )
        }
    
    def _get_system_template(self) -> str:
        """Get the system prompt template."""
        return """You are a professional pandas DataFrame assistant operating in a {security_level} environment.\n\nYour ONLY source of truth is a DataFrame named `df`.\n\nDataFrame schema:\n{schema}\n\nColumn data types:\n{dtypes}\n\nSample rows:\n{sample_rows}\n\nColumn value examples:\n{examples_text}\n\n{security_rules}\n\n{langgraph_rules}\n\n{output_format_rules}\n\n{code_style_rules}\n\n{error_handling_rules}"""

    def _get_query_template(self) -> str:
        """Get the query execution prompt template."""
        return """You are a precision-focused pandas code assistant operating in a {security_level} data environment.\n\nYour exclusive data source is a single DataFrame named `df`.\n\nAvailable columns: {columns}\nColumn data types: {dtypes}\nSample rows:\n{sample_rows}\nColumn value examples:\n{examples_text}\nDataFrame shape: {shape}\nOutput format: {output_format}\n\n{data_source_rules}\n\n{column_enforcement_rules}\n\n{syntax_compliance_rules}\n\n{string_filtering_rules}\n\n{ranking_queries_rules}\n\n{output_policy_rules}\n\n{plotting_restrictions}\n\n{strict_prohibitions}\n\nQuestion: {query}"""

    def _get_plot_template(self) -> str:
        """Get the plot creation prompt template."""
        return """Create a {plot_type} visualization with the following specifications:

X-axis column: {x_column}
Y-axis column: {y_column}
Title: {title}
Color grouping: {color_column}

{plot_guidelines}

Use the create_plot tool with these exact parameters."""

    def _get_error_correction_template(self) -> str:
        """Get the error correction prompt template."""
        return """The previous query failed with error: {error_message}

Original query: {original_query}

{correction_guidelines}

Please provide a corrected version that addresses the error."""

    def _get_context_aware_template(self) -> str:
        """Get the context-aware prompt template."""
        return """# CONTEXT: df refers to rows where {context_description}

{query}

{context_guidelines}

Please query the filtered DataFrame with this context in mind."""

    def _get_column_suggestion_template(self) -> str:
        """Get the column suggestion prompt template."""
        return """Query: {query}

Available columns: {available_columns}

{column_guidelines}

Please use the exact column names from the available columns list."""

    def _get_aggregation_template(self) -> str:
        """Get the aggregation prompt template."""
        return """Calculate the {operation} of {column}

{aggregation_guidelines}

Provide the result in the specified format."""

    def _get_filtering_template(self) -> str:
        """Get the filtering prompt template."""
        return """Filter rows where {column} {operation} {value}

{filtering_guidelines}

Use the appropriate filtering syntax for the operation."""

    def _get_data_exploration_template(self) -> str:
        """Get the data exploration prompt template."""
        return """Provide a comprehensive overview of the DataFrame including:

{exploration_guidelines}

Focus on actionable insights and data quality assessment."""

    def _get_complex_query_template(self) -> str:
        """Get the complex query prompt template."""
        return """Execute a complex query with multiple parts:

Query parts: {query_parts}
Combine operation: {combine_operation}

{complex_query_guidelines}

Execute each part separately and combine results appropriately."""

    def _get_query_analysis_template(self) -> str:
        """Get the query analysis prompt template."""
        return """Analyze the user query to determine the appropriate tool. First, provide a brief step-by-step reasoning for your decision. Then, provide a JSON object with the final answer.

Available columns: {columns}
User Query: "{query}"

**Reasoning Steps:**
1.  **Identify Intent**: Is the user asking for a specific piece of data, a list, a ranking (e.g., "top 5"), or an aggregation (e.g., "count")? Or are they asking for a visual representation of data?
2.  **Check for Keywords**: Look for explicit visualization keywords (e.g., "plot", "chart", "graph") versus data analysis keywords (e.g., "top", "bottom", "count", "average").
3.  **Decision**: Based on the intent and keywords, decide if a plot is required. Queries for rankings like "top 5" are data analysis unless a plot is explicitly requested.
4.  **Parameters**: Determine plot parameters (`plot_type`, `x_column`, `y_column`) only if a plot is required.

**JSON Output:**
Your task is to return a JSON object with the following structure:
{{
  "should_plot": boolean,
  "confidence": float,
  "suggested_tool": "create_plot" or "query_df",
  "plot_type": "string or null",
  "x_column": "string or null",
  "y_column": "string or null"
}}
"""

    def _get_complex_query_parser_template(self) -> str:
        """Get the complex query parser prompt template."""
        return """Analyze the following user query and break it down into a JSON array of simple, standalone sub-queries. Each sub-query should be a complete, executable question.

User Query: "{query}"

Return a JSON object with a single key "sub_queries" containing the array of strings.

Examples:
- "what are the top 2 rent in mumbai, bottom 3 in noida and average in delhi and plot pie chart for gender in indore" -> {{"sub_queries": ["what are the top 2 rent in mumbai", "what are the bottom 3 rent in noida", "what is the average rent in delhi", "plot a pie chart for gender in indore"]}}
- "show me the total count of employees and also a bar chart of their salaries" -> {{"sub_queries": ["show me the total count of employees", "show me a bar chart of their salaries"]}}
- "how many people work in the IT department vs the HR department" -> {{"sub_queries": ["how many people work in the IT department", "how many people work in the HR department"]}}
"""

    def _get_security_rules(self) -> str:
        """Get security rules based on environment."""
        security_level = self.config.security_config.security_level
        
        if security_level == "production":
            return """CRITICAL SECURITY RULES (MUST be followed — violations may result in data breach, lawsuits, or financial loss):

1. You MUST use only the DataFrame `df`. Do NOT use, assume, reference, or generate data from any other source under any condition.

2. You MUST use only the exact column names from the schema above. Do NOT guess, infer, correct, or hallucinate column names.

3. You MUST NOT use or generate:
   - `pd.DataFrame(...)` or any custom/sample/fake dataset
   - `df.sample()` or other sample generation
   - Any fabricated, invented, or hypothetical values or categories
   - Any values not already present in `df`

4. If a requested column does not exist in `df`, return:
print("Column not found in DataFrame")

5. Violating any of the above rules could result in irreversible harm, including legal action, regulatory violation, or financial loss."""
        
        elif security_level == "staging":
            return """SECURITY RULES (Staging Environment):

1. Use only the DataFrame `df` as your data source.
2. Use exact column names from the schema.
3. Avoid generating fake or sample data.
4. Return "Column not found in DataFrame" for missing columns.
5. Some restrictions may be relaxed for testing purposes."""
        
        else:  # development
            return """DEVELOPMENT RULES:

1. Use the DataFrame `df` as your primary data source.
2. Use column names from the schema when possible.
3. Some flexibility allowed for testing and development."""

    def _get_langgraph_rules(self) -> str:
        """Get LangGraph workflow rules."""
        return """LANGGRAPH WORKFLOW RULES:

The system uses a sophisticated LangGraph conditional workflow that automatically:
- Analyzes your query intent
- Routes to the appropriate tool (data analysis or visualization)
- Executes the query with the right approach
- Formats the response appropriately

AVAILABLE TOOLS:
- query_df: For data analysis queries (count, average, filter, etc.)
- create_plot: For visualization requests (plot, chart, graph, etc.)

TOOL USAGE GUIDELINES:

✅ Use query_df for:
- "count", "number", "how many", "total", "sum", "average", "mean"
- "filter", "find", "search", "show", "list", "get", "extract"
- "what is", "what are", "which", "where", "when", "who"
- ANY query about data analysis WITHOUT visualization keywords

✅ Use create_plot for:
- "plot", "chart", "graph", "visualization", "visualize"
- "histogram", "bar chart", "line chart", "scatter plot"
- "show me a chart", "create a graph", "display a plot"
- EXPLICIT visualization requests

❌ NEVER use create_plot for:
- Simple counting queries
- Data analysis without visualization keywords
- When user asks for numbers, counts, or statistics only

EXAMPLES:
✅ "How many records are there?" → Use query_df
✅ "What is the average salary?" → Use query_df  
✅ "Show me a bar chart of salaries" → Use create_plot
✅ "Plot the distribution of ages" → Use create_plot
❌ "Count the records" → Use query_df (NOT create_plot)

REMEMBER:
- The LangGraph workflow handles routing automatically
- Only create plots when users EXPLICITLY ask for visualizations"""

    def _get_output_format_rules(self) -> str:
        """Get output format rules based on configuration."""
        output_format = self.config.data_config.default_output_format
        
        if output_format == "table":
            return """OUTPUT FORMAT: Return results in a well-formatted table with clear headers and aligned columns."""
        elif output_format == "string":
            return """OUTPUT FORMAT: Return results as a clear, concise string summary."""
        elif output_format == "summary":
            return """OUTPUT FORMAT: Return results as a comprehensive summary with key insights and statistics."""
        else:  # auto
            return """OUTPUT FORMAT: Automatically choose the most appropriate format (table, string, or summary) based on the query type and result complexity."""

    def _get_code_style_rules(self) -> str:
        """Get code style and syntax rules."""
        return """CODE STYLE RULES:

1. For multiple columns, use:
df[['col1', 'col2']]
NOT:
df[('col1', 'col2')]

2. For groupby with multiple columns, use:
df.groupby(['col1', 'col2'])
NOT:
df.groupby(('col1', 'col2'))

3. Always include na=False in string operations to handle missing data safely.

4. Ensure ALL brackets, parentheses, and quotes are properly closed.

5. Generate COMPLETE, VALID Python code that can execute without syntax errors."""

    def _get_error_handling_rules(self) -> str:
        """Get error handling rules."""
        return """ERROR HANDLING:

1. If a requested column does not exist in `df`, return:
print("Column not found in DataFrame")

2. If no data matches the filter criteria, return:
print("No data found matching the criteria")

3. For empty results, provide informative messages rather than empty outputs.

4. Handle missing values appropriately with na=False in string operations."""

    def _get_data_source_rules(self) -> str:
        """Get data source constraint rules."""
        return """DATA SOURCE CONSTRAINT:
- Use only the `df` DataFrame. Do NOT create, simulate, sample, or infer data:
  No `pd.DataFrame(...)`, `df.sample()`, dummy rows, or assumptions."""

    def _get_column_enforcement_rules(self) -> str:
        """Get column name enforcement rules."""
        return """COLUMN NAME ENFORCEMENT:
- Use only the exact column names listed above.
- If a column is not found, return:
print("Column not found in DataFrame")"""

    def _get_syntax_compliance_rules(self) -> str:
        """Get syntax compliance rules."""
        return """SYNTAX COMPLIANCE:
- Access multiple columns with:
df[['col1', 'col2']]
- Use groupby with:
df.groupby(['col1', 'col2'])
- Never use tuple-style indexing like df[('col1', 'col2')]

DATA TYPE AWARENESS:
- For NUMERIC columns: Use df.nlargest(N, 'column') for top N
- For STRING/CATEGORICAL columns: Use df['column'].value_counts().head(N) for top N
- For counting categories: Use df['column'].value_counts()
- For numeric operations: Use df['column'].sum(), df['column'].mean(), etc.
- For string operations: Use df['column'].str.contains(), df['column'].str.lower(), etc."""

    def _get_string_filtering_rules(self) -> str:
        """Get string filtering protocols."""
        return """STRING FILTERING PROTOCOLS:
- For partial match:
df[df['column'].str.lower().str.strip().str.contains('value', case=False, na=False)]
- For exact match:
df[df['column'].str.lower().str.strip() == 'value']
- Always include na=False for null-safe operations.

FILTERED COUNT FORMAT:
df[df['column'].str.lower().str.strip().str.contains('value', case=False, na=False)].shape[0]"""

    def _get_ranking_queries_rules(self) -> str:
        """Get ranking queries rules."""
        return """TOP/BOTTOM/BEST QUERIES:
- For "top N" queries on NUMERIC columns: df.nlargest(N, 'column')
- For "bottom N" queries on NUMERIC columns: df.nsmallest(N, 'column')
- For "top N" queries on CATEGORICAL/STRING columns: df['column'].value_counts().head(N)
- For "best N" queries on NUMERIC columns: df.nlargest(N, 'column') (same as top)
- For filtered top/bottom/best on NUMERIC: df[df['filter_column'] == 'value'].nlargest(N, 'sort_column')
- For filtered top/bottom/best on CATEGORICAL: df[df['filter_column'] == 'value']['column'].value_counts().head(N)

IMPORTANT: Use value_counts().head(N) for string/categorical columns, nlargest(N, 'column') for numeric columns only."""

    def _get_output_policy_rules(self) -> str:
        """Get output policy rules."""
        return """OUTPUT POLICY:
- Return only one executable line starting with df[...] or print(...).
- Do NOT return markdown, comments, explanations, or logging.
- Ensure ALL brackets, parentheses, and quotes are properly closed.
- Generate COMPLETE, VALID Python code that can execute without syntax errors."""

    def _get_plotting_restrictions(self) -> str:
        """Get plotting restrictions."""
        return """PLOTTING RESTRICTION:
- Do NOT use matplotlib, seaborn, or any libraries.
- Use ONLY the `create_plot` tool if plotting is required.
- ONLY create plots when user EXPLICITLY asks for plots/charts/graphs/visualizations
- For simple queries like "count", "average", "how many", use query_df only"""

    def _get_strict_prohibitions(self) -> str:
        """Get strict prohibitions."""
        return """STRICT PROHIBITIONS:
- No placeholder, fabricated, or guessed data.
- No assumption or correction of column names.
- No response format other than final valid Python code using df."""

    def _get_plot_guidelines(self) -> str:
        """Get plot creation guidelines."""
        return """PLOT GUIDELINES:
- Choose appropriate plot type based on data characteristics
- Ensure axes are properly labeled
- Use meaningful titles
- Consider color schemes for better readability
- Handle missing data appropriately"""

    def _get_correction_guidelines(self) -> str:
        """Get error correction guidelines."""
        return """CORRECTION GUIDELINES:
- Identify the specific error in the original code
- Provide a corrected version that addresses the root cause
- Ensure the corrected code follows all syntax and security rules
- Test the logic to prevent similar errors"""

    def _get_context_guidelines(self) -> str:
        """Get context-aware query guidelines."""
        return """CONTEXT GUIDELINES:
- Consider the filtered DataFrame context when writing queries
- Ensure column references are valid within the filtered dataset
- Apply filters appropriately to the context
- Maintain data integrity within the filtered scope"""

    def _get_column_guidelines(self) -> str:
        """Get column suggestion guidelines."""
        return """COLUMN GUIDELINES:
- Use exact column names from the available list
- Consider fuzzy matching for similar column names
- Validate column existence before using
- Provide helpful suggestions for missing columns"""

    def _get_aggregation_guidelines(self) -> str:
        """Get aggregation guidelines."""
        return """AGGREGATION GUIDELINES:
- Choose appropriate aggregation function for the data type
- Handle missing values appropriately
- Consider grouping when relevant
- Provide meaningful result descriptions"""

    def _get_filtering_guidelines(self) -> str:
        """Get filtering guidelines."""
        return """FILTERING GUIDELINES:
- Use appropriate comparison operators
- Handle string vs numeric comparisons correctly
- Consider case sensitivity for string filters
- Apply multiple filters when needed"""

    def _get_exploration_guidelines(self) -> str:
        """Get data exploration guidelines."""
        return """EXPLORATION GUIDELINES:
1. Basic statistics (shape, data types, missing values)
2. Summary statistics for numeric columns
3. Value counts for categorical columns
4. Data quality assessment
5. Notable patterns or insights
6. Recommendations for further analysis"""

    def _get_complex_query_guidelines(self) -> str:
        """Get complex query guidelines."""
        return """COMPLEX QUERY GUIDELINES:
- Execute each query part separately
- Combine results using the specified operation
- Maintain data consistency across parts
- Handle edge cases appropriately
- Provide clear result organization"""

    # Public methods for backward compatibility and ease of use
    def get_system_prompt(self, df: pd.DataFrame, examples_text: str = "") -> str:
        """Get the main system prompt for the LangGraph agent, dynamically using the DataFrame schema."""
        schema = ', '.join([f"{col}" for col in df.columns])
        dtypes = ', '.join([f"{col}: {str(dtype)}" for col, dtype in df.dtypes.items()])
        # --- Enhanced: Show unique values for categorical, sample/min/max for numeric ---
        col_examples = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                values = df[col].dropna().unique()
                sample_vals = list(values[:5])
                min_val = df[col].min() if not df[col].empty else None
                max_val = df[col].max() if not df[col].empty else None
                mean_val = df[col].mean() if not df[col].empty else None
                summary = f"Sample: {sample_vals}"
                if min_val is not None and max_val is not None:
                    summary += f", Min: {min_val}, Max: {max_val}"
                if mean_val is not None:
                    summary += f", Mean: {mean_val:.2f}"
                col_examples.append(f"{col}: {summary}")
            else:
                unique_vals = df[col].dropna().unique()
                unique_vals = list(unique_vals[:20])  # Limit to 20 for brevity
                col_examples.append(f"{col}: Unique values: {unique_vals}")
        examples_text = '\n'.join(col_examples) if col_examples else "No value examples available."
        sample_rows = None  # Not used anymore
        # Get all the rule components
        security_rules = self._get_security_rules()
        langgraph_rules = self._get_langgraph_rules()
        output_format_rules = self._get_output_format_rules()
        code_style_rules = self._get_code_style_rules()
        error_handling_rules = self._get_error_handling_rules()
        return self.templates['system'].format(
            schema=schema,
            dtypes=dtypes,
            sample_rows="",  # No sample rows, replaced by examples_text
            examples_text=examples_text,
            security_level=self.config.security_config.security_level,
            security_rules=security_rules,
            langgraph_rules=langgraph_rules,
            output_format_rules=output_format_rules,
            code_style_rules=code_style_rules,
            error_handling_rules=error_handling_rules
        )

    def get_query_prompt(self, query: str, df: pd.DataFrame, output_format: str = None) -> str:
        """Get the query execution prompt for the QueryExecutor, dynamically using the DataFrame schema."""
        columns = ', '.join(df.columns)
        dtypes = ', '.join([f"{col}: {str(dtype)}" for col, dtype in df.dtypes.items()])
        
        # --- Enhanced: Show unique values for categorical, sample/min/max for numeric ---
        col_examples = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                values = df[col].dropna().unique()
                sample_vals = list(values[:5])
                min_val = df[col].min() if not df[col].empty else None
                max_val = df[col].max() if not df[col].empty else None
                mean_val = df[col].mean() if not df[col].empty else None
                summary = f"Sample: {sample_vals}"
                if min_val is not None and max_val is not None:
                    summary += f", Min: {min_val}, Max: {max_val}"
                if mean_val is not None:
                    summary += f", Mean: {mean_val:.2f}"
                col_examples.append(f"{col} (numeric): {summary}")
            else:
                unique_vals = df[col].dropna().unique()
                unique_vals = list(unique_vals[:20])  # Limit to 20 for brevity
                col_examples.append(f"{col} (categorical): Unique values: {unique_vals}")
        examples_text = '\n'.join(col_examples) if col_examples else "No value examples available."
        
        shape = df.shape
        output_format = output_format or self.config.data_config.default_output_format
        
        return self.templates['query'].format(
            query=query,
            columns=columns,
            dtypes=dtypes,
            sample_rows="",  # No sample rows, replaced by examples_text
            examples_text=examples_text,
            shape=shape,
            output_format=output_format,
            security_level=self.config.security_config.security_level,
            data_source_rules=self._get_data_source_rules(),
            column_enforcement_rules=self._get_column_enforcement_rules(),
            syntax_compliance_rules=self._get_syntax_compliance_rules(),
            string_filtering_rules=self._get_string_filtering_rules(),
            ranking_queries_rules=self._get_ranking_queries_rules(),
            output_policy_rules=self._get_output_policy_rules(),
            plotting_restrictions=self._get_plotting_restrictions(),
            strict_prohibitions=self._get_strict_prohibitions()
        )

    def get_plot_creation_prompt(self, plot_type: str, x_column: str = None, y_column: str = None, 
                                title: str = None, color_column: str = None) -> str:
        """Get prompt for plot creation guidance."""
        if plot_type is None:
            raise ValueError("Error: plot_type is None in get_plot_creation_prompt. Please specify a valid plot type.")
        return self.templates['plot'].format(
            plot_type=plot_type,
            x_column=x_column or 'Not specified',
            y_column=y_column or 'Not specified',
            title=title or f'{plot_type.title()} Plot',
            color_column=color_column or 'None',
            plot_guidelines=self._get_plot_guidelines()
        )

    def get_error_correction_prompt(self, original_query: str, error_message: str, 
                                  suggested_column: str = None) -> str:
        """Get prompt for error correction scenarios."""
        prompt = self.templates['error_correction'].format(
            original_query=original_query,
            error_message=error_message,
            correction_guidelines=self._get_correction_guidelines()
        )
        
        if suggested_column:
            prompt += f"\n\nNote: Did you mean to use column '{suggested_column}'?"
        
        return prompt

    def get_context_aware_prompt(self, query: str, context_description: str) -> str:
        """Get prompt for context-aware queries."""
        return self.templates['context_aware'].format(
            query=query,
            context_description=context_description,
            context_guidelines=self._get_context_guidelines()
        )

    def get_column_suggestion_prompt(self, query: str, available_columns: list, 
                                   suggested_columns: list = None) -> str:
        """Get prompt for column suggestion scenarios."""
        prompt = self.templates['column_suggestion'].format(
            query=query,
            available_columns=', '.join(available_columns),
            column_guidelines=self._get_column_guidelines()
        )
        
        if suggested_columns:
            prompt += f"\n\nSuggested columns for your query: {', '.join(suggested_columns)}"
        
        return prompt

    def get_aggregation_prompt(self, operation: str, column: str, group_by: str = None) -> str:
        """Get prompt for aggregation operations."""
        prompt = self.templates['aggregation'].format(
            operation=operation,
            column=column,
            aggregation_guidelines=self._get_aggregation_guidelines()
        )
        
        if group_by:
            prompt += f" grouped by {group_by}"
        
        return prompt

    def get_filtering_prompt(self, column: str, value: str, operation: str = "equals") -> str:
        """Get prompt for filtering operations."""
        return self.templates['filtering'].format(
            column=column,
            value=value,
            operation=operation,
            filtering_guidelines=self._get_filtering_guidelines()
        )

    def get_data_exploration_prompt(self) -> str:
        """Get prompt for general data exploration."""
        return self.templates['data_exploration'].format(
            exploration_guidelines=self._get_exploration_guidelines()
        )

    def get_complex_query_prompt(self, query_parts: List[str], combine_operation: str = "and") -> str:
        """Get prompt for complex queries with multiple parts."""
        return self.templates['complex_query'].format(
            query_parts=query_parts,
            combine_operation=combine_operation,
            complex_query_guidelines=self._get_complex_query_guidelines()
        )

    def get_visualization_suggestion_prompt(self, data_summary: str) -> str:
        """Get prompt for suggesting visualizations based on data."""
        return f"""Based on this data summary:
{data_summary}

Please suggest appropriate visualizations that would be most informative for understanding this data."""

    def get_custom_prompt(self, template_name: str, **kwargs) -> str:
        """Get a custom prompt using a template with keyword arguments."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available templates: {list(self.templates.keys())}")
        
        return self.templates[template_name].format(**kwargs)

    def add_custom_template(self, name: str, template: str, required_vars: List[str] = None):
        """Add a custom prompt template."""
        self.templates[name] = PromptTemplate(template, required_vars)
        logger.info(f"Added custom prompt template: {name}")

    def get_template_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available templates."""
        return {
            name: {
                'required_vars': template.required_vars,
                'template_preview': template.template[:100] + '...' if len(template.template) > 100 else template.template
            }
            for name, template in self.templates.items()
        }

    def validate_template(self, template_name: str, **kwargs) -> bool:
        """Validate that a template can be formatted with the provided variables."""
        try:
            self.get_custom_prompt(template_name, **kwargs)
            return True
        except Exception as e:
            logger.warning(f"Template validation failed for '{template_name}': {e}")
            return False

    @classmethod
    def get_all_prompts(cls) -> Dict[str, str]:
        """Get all available prompts for reference."""
        instance = cls()
        return {
            "system_prompt": instance.get_system_prompt(pd.DataFrame()),
            "query_prompt": instance.get_query_prompt("", pd.DataFrame(), "table"),
            "plot_creation_prompt": instance.get_plot_creation_prompt("bar"),
            "error_correction_prompt": instance.get_error_correction_prompt("", ""),
            "context_aware_prompt": instance.get_context_aware_prompt("", ""),
            "column_suggestion_prompt": instance.get_column_suggestion_prompt("", []),
            "aggregation_prompt": instance.get_aggregation_prompt("mean", "column"),
            "filtering_prompt": instance.get_filtering_prompt("column", "value"),
            "data_exploration_prompt": instance.get_data_exploration_prompt(),
            "complex_query_prompt": instance.get_complex_query_prompt(["part1", "part2"])
        }

    @staticmethod
    def get_system_prompt(schema: str, examples_text: str) -> str:
        """Static method for backward compatibility."""
        instance = PromptManager()
        return instance._get_system_prompt_static(schema, examples_text)
    
    def _get_system_prompt_static(self, schema: str, examples_text: str) -> str:
        """Internal method to avoid recursion."""
        # Get all the rule components
        security_rules = self._get_security_rules()
        langgraph_rules = self._get_langgraph_rules()
        output_format_rules = self._get_output_format_rules()
        code_style_rules = self._get_code_style_rules()
        error_handling_rules = self._get_error_handling_rules()
        
        return self.templates['system'].format(
            schema=schema,
            examples_text=examples_text,
            security_level=self.config.security_config.security_level,
            security_rules=security_rules,
            langgraph_rules=langgraph_rules,
            output_format_rules=output_format_rules,
            code_style_rules=code_style_rules,
            error_handling_rules=error_handling_rules
        )
