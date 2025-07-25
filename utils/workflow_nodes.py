import pandas as pd
from utils.logger import setup_logger
from utils.query_executor import QueryExecutor
from utils.plot_manager import PlotManager

logger = setup_logger(__name__)

def query_df_node(state: dict, query_executor: QueryExecutor) -> dict:
    """
    Node function for querying the DataFrame using pandas safely with retry mechanism.
    Args:
        state: The current workflow state (dict)
        query_executor: QueryExecutor instance
    Returns:
        Updated state with 'tool_result' set to the query result
    """
    query = state.get("query") or state.get("user_input")
    logger.debug(f"[LangGraph Node] Executing query_df_node with query: {query}")
    formatted_result, raw_result = query_executor.execute_query_with_retry(query)
    state["tool_result"] = formatted_result
    state["raw_tool_result"] = raw_result
    return state

def create_plot_node(state: dict, plot_manager: PlotManager, df: pd.DataFrame) -> dict:
    """
    Node function for creating plots using matplotlib and seaborn.
    Args:
        state: The current workflow state (dict)
        plot_manager: PlotManager instance
        df: The DataFrame to plot from
    Returns:
        Updated state with 'tool_result' set to the plot creation status
    """
    plot_type = state.get("plot_type")
    x_column = state.get("x_column")
    y_column = state.get("y_column")
    title = state.get("title")
    color_column = state.get("color_column")
    additional_params = state.get("additional_params")
    logger.debug(f"[LangGraph Node] Executing create_plot_node with type: {plot_type}")
    result = plot_manager.create_plot(
        df, plot_type, x_column, y_column, title, color_column, additional_params
    )
    state["tool_result"] = result
    return state
