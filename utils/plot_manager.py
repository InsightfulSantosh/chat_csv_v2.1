import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.config_manager import get_config_manager
from utils.logger import setup_logger
from utils.fuzzy_matcher import FuzzyMatcher
from utils.prompts import PromptManager

logger = setup_logger(__name__)

class PlotManager:
    """Handles all plotting functionality for the SmartPandasAgent."""
    
    def __init__(self, fuzzy_matcher: FuzzyMatcher):
        self.fuzzy_matcher = fuzzy_matcher
        self.current_plot = None
        self.prompt_manager = PromptManager()
        
        # Set matplotlib to non-interactive backend
        plt.switch_backend('Agg')
    
    def _validate_column(self, column: str, df: pd.DataFrame) -> str:
        """Validate and suggest column names using fuzzy matching."""
        if column and column not in df.columns:
            suggested = self.fuzzy_matcher.fuzzy_match_column(column)
            if suggested:
                return suggested
            else:
                raise ValueError(f"Column '{column}' not found. Available columns: {', '.join(df.columns)}")
        return column
    
    def _create_seaborn_plot(self, plot_type: str, plot_df: pd.DataFrame, 
                           x_column: str, y_column: str, color_column: str, ax) -> str:
        """
        Create seaborn plots with consistent parameter handling.
        
        Args:
            plot_type: Type of seaborn plot (barplot, lineplot, scatterplot, etc.)
            plot_df: DataFrame to plot
            x_column: X-axis column
            y_column: Y-axis column
            color_column: Color grouping column
            ax: Matplotlib axis object
            
        Returns:
            "success" if plot created successfully, error message otherwise
        """
        try:
            plot_func = getattr(sns, plot_type)
            kwargs = {'data': plot_df, 'ax': ax}
            
            if x_column:
                kwargs['x'] = x_column
            if y_column:
                kwargs['y'] = y_column
            if color_column:
                kwargs['hue'] = color_column
                
            plot_func(**kwargs)
            return "success"
        except Exception as e:
            return f"Error creating {plot_type}: {str(e)}"
    
    def create_plot(self, df: pd.DataFrame, plot_type: str = None, x_column: str = None, 
                   y_column: str = None, title: str = None, color_column: str = None,
                   type: str = None, additional_params: str = None) -> str:
        """Create various types of plots using matplotlib and seaborn."""
        plot_type = plot_type or type
        if plot_type is None:
            return ("Error: Could not determine plot type from your query. "
                    "Please specify the type of plot you want (e.g., histogram, bar, line, etc.).")
        try:
            # Use plot creation prompt for better guidance
            plot_prompt = self.prompt_manager.get_plot_creation_prompt(
                plot_type=plot_type,
                x_column=x_column,
                y_column=y_column,
                title=title,
                color_column=color_column
            )
            logger.debug(f"Plot creation prompt: {plot_prompt}")
            
            # Clear previous plots
            plt.clf()
            plt.close('all')

            # Set style
            config = get_config_manager()
            plt.style.use(config.plot_config.plot_style)
            sns.set_palette(config.plot_config.seaborn_palette)

            # Create figure with good size
            fig, ax = plt.subplots(figsize=config.plot_config.default_plot_size)

            # Validate columns
            try:
                x_column = self._validate_column(x_column, df)
                y_column = self._validate_column(y_column, df)
                color_column = self._validate_column(color_column, df)
            except ValueError as e:
                return str(e)

            # Filter out NaN values for plotting
            plot_df = df.copy()
            if x_column:
                plot_df = plot_df.dropna(subset=[x_column])
            if y_column:
                plot_df = plot_df.dropna(subset=[y_column])

            if plot_df.empty:
                return "No data available for plotting after removing NaN values."

            # Create plots based on type
            plot_result = self._create_specific_plot(plot_df, plot_type, x_column, y_column, 
                                                   color_column, ax)
            
            if plot_result != "success":
                return plot_result

            # Set title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"{plot_type.title()} Plot")

            # Rotate x-axis labels if they're too long
            if x_column and not pd.api.types.is_numeric_dtype(plot_df[x_column]):
                ax.tick_params(axis='x', rotation=config.plot_config.x_axis_label_rotation)

            # Adjust layout
            plt.tight_layout()

            # Store the current plot
            self.current_plot = fig

            return f"✅ {plot_type.title()} plot created successfully! Use get_plot() to retrieve it."

        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            return f"Error creating plot: {str(e)}"

    def _create_specific_plot(self, plot_df: pd.DataFrame, plot_type: str, 
                            x_column: str, y_column: str, color_column: str, ax) -> str:
        """Create specific plot types."""
        plot_type_lower = plot_type.lower()
        
        if plot_type_lower == 'bar':
            return self._create_bar_plot(plot_df, x_column, y_column, color_column, ax)
        elif plot_type_lower == 'line':
            return self._create_line_plot(plot_df, x_column, y_column, color_column, ax)
        elif plot_type_lower == 'scatter':
            return self._create_scatter_plot(plot_df, x_column, y_column, color_column, ax)
        elif plot_type_lower == 'histogram':
            return self._create_histogram_plot(plot_df, x_column, color_column, ax)
        elif plot_type_lower == 'box':
            return self._create_box_plot(plot_df, x_column, y_column, color_column, ax)
        elif plot_type_lower in ['pie', 'piechart']:
            return self._create_pie_plot(plot_df, x_column, y_column, ax)
        elif plot_type_lower == 'heatmap':
            return self._create_heatmap_plot(plot_df, x_column, y_column, color_column, ax)
        elif plot_type_lower == 'countplot':
            return self._create_count_plot(plot_df, x_column, color_column, ax)
        else:
            return f"Unsupported plot type: {plot_type}. Supported types: bar, line, scatter, histogram, box, pie, heatmap, countplot"

    def _create_bar_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, 
                        color_column: str, ax) -> str:
        """Create bar plot."""
        if y_column:
            # Grouped bar plot
            return self._create_seaborn_plot('barplot', plot_df, x_column, y_column, color_column, ax)
        else:
            # Count plot
            return self._create_seaborn_plot('countplot', plot_df, x_column, None, color_column, ax)

    def _create_line_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, 
                         color_column: str, ax) -> str:
        """Create line plot."""
        if y_column:
            return self._create_seaborn_plot('lineplot', plot_df, x_column, y_column, color_column, ax)
        else:
            return "Line plot requires both x and y columns."

    def _create_scatter_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, 
                           color_column: str, ax) -> str:
        """Create scatter plot."""
        if y_column:
            return self._create_seaborn_plot('scatterplot', plot_df, x_column, y_column, color_column, ax)
        else:
            return "Scatter plot requires both x and y columns."

    def _create_histogram_plot(self, plot_df: pd.DataFrame, x_column: str, 
                             color_column: str, ax) -> str:
        """Create histogram plot."""
        config = get_config_manager()
        if color_column:
            for category in plot_df[color_column].unique():
                subset = plot_df[plot_df[color_column] == category]
                ax.hist(subset[x_column], bins=config.plot_config.histogram_bins, alpha=config.plot_config.histogram_alpha, label=category)
            ax.legend()
        else:
            ax.hist(plot_df[x_column], bins=config.plot_config.histogram_bins, alpha=config.plot_config.histogram_alpha)
        return "success"

    def _create_box_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, 
                        color_column: str, ax) -> str:
        """Create box plot."""
        if y_column:
            return self._create_seaborn_plot('boxplot', plot_df, x_column, y_column, color_column, ax)
        else:
            # For box plot without y_column, use y=x_column
            return self._create_seaborn_plot('boxplot', plot_df, None, x_column, color_column, ax)

    def _create_pie_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, ax) -> str:
        """Create pie plot."""
        logger.info(f"Creating pie plot with x_column='{x_column}' and y_column='{y_column}'")
        
        if isinstance(plot_df, pd.Series):
            plot_df = plot_df.to_frame()

        logger.info(f"DataFrame shape: {plot_df.shape}")
        logger.info(f"DataFrame columns: {plot_df.columns.tolist()}")
        logger.info(f"DataFrame head:\n{plot_df.head()}")

        if x_column is None:
            if plot_df.shape[1] == 1:
                x_column = plot_df.columns[0]
            else:
                return "Pie chart requires a column for the labels (x_column)."
            
        if y_column:
            pie_data = plot_df.groupby(x_column)[y_column].sum()
        else:
            pie_data = plot_df[x_column].value_counts()

        if pie_data.empty:
            return "No data available for pie chart."

        logger.info(f"Pie data:\n{pie_data}")
        config = get_config_manager()
        ax.pie(pie_data.values, labels=pie_data.index, autopct=config.plot_config.pie_percent_format)
        return "success"

    def _create_heatmap_plot(self, plot_df: pd.DataFrame, x_column: str, y_column: str, 
                           color_column: str, ax) -> str:
        """Create heatmap plot."""
        config = get_config_manager()
        if y_column:
            # Create pivot table for heatmap
            pivot_data = plot_df.pivot_table(values=y_column, index=x_column,
                                            columns=color_column if color_column else None,
                                            aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, fmt=config.plot_config.heatmap_float_format, ax=ax)
        else:
            # Correlation heatmap
            numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = plot_df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt=config.plot_config.heatmap_float_format, ax=ax)
            else:
                return "Not enough numeric columns for correlation heatmap."
        return "success"

    def _create_count_plot(self, plot_df: pd.DataFrame, x_column: str, 
                          color_column: str, ax) -> str:
        """Create count plot."""
        return self._create_seaborn_plot('countplot', plot_df, x_column, None, color_column, ax)

    def get_plot(self):
        """Get the current plot figure."""
        return self.current_plot

    def save_plot(self, filename: str, dpi: int = None):
        """Save the current plot to a file."""
        config = get_config_manager()
        if dpi is None:
            dpi = config.plot_config.default_dpi
            
        if self.current_plot:
            self.current_plot.savefig(filename, dpi=dpi, bbox_inches='tight')
            return f"Plot saved to {filename}"
        else:
            return "No plot available to save."

    def clear_plot(self):
        """Clear the current plot."""
        self.current_plot = None
        plt.clf()
        plt.close('all')
