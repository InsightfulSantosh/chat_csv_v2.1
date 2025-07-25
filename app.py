import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
import tempfile
import shutil
import logging
import atexit
from main_orchestrator import SmartPandasAgent
from utils.config_manager import get_config_manager
from utils.langsmith_dashboard import render_langsmith_status, create_langsmith_tab
from utils.logger import setup_logger
from utils.prompts import PromptManager
from utils.data_loader import load_csv_with_preprocessing, validate_csv_path
from utils.question_parser import rewrite_user_question
from utils.langsmith_config import get_langsmith_manager
import numpy as np
from scipy.stats import skew, kurtosis
import re

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="SmartPandasAgent - Conversational Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #1f77b4;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #28a745;
    }
    .error-message {
        background-color: #ffe6e6;
        border-left-color: #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .plot-container {
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'csv_path' not in st.session_state:
        st.session_state.csv_path = None
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'developer_mode' not in st.session_state:
        st.session_state.developer_mode = False
    if 'prompt_manager' not in st.session_state:
        st.session_state.prompt_manager = PromptManager()
    if 'output_format' not in st.session_state:
        st.session_state.output_format = 'plain'

def load_data(csv_path, is_uploaded_file=False):
    """Load and validate CSV data with production-grade preprocessing."""
    try:
        # Validate the CSV file first
        if not validate_csv_path(csv_path):
            return None, "Invalid or inaccessible CSV file"
        
        # Determine output folder based on file type
        if is_uploaded_file:
            # For uploaded files, use temp directory
            output_folder = st.session_state.temp_dir
        else:
            # For default files, use data/formated_data
            output_folder = "data/formated_data"
        
        # Use production-grade data preprocessing
        df = load_csv_with_preprocessing(
            input_path=csv_path,
            lowercase=True,  # Convert column names and values to lowercase
            rename_columns=None,  # No custom renames for uploaded files
            save_formatted=True,  # Save formatted version for consistency
            output_folder=output_folder
        )
        
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def initialize_agent(csv_path, provider=None, model=None, temperature=None):
    """Initialize the SmartPandasAgent with the given parameters."""
    try:
        config = get_config_manager()
        agent = SmartPandasAgent(
            csv_path=csv_path,
            provider=provider or config.llm_config.default_provider,
            model=model or config.llm_config.default_model,
            temperature=temperature or config.llm_config.default_temperature
        )
        return agent, None
    except Exception as e:
        return None, str(e)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for display."""
    if fig is None:
        return None
    
    # Save figure to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    
    return img_str

def display_plot(fig):
    """Display plot in Streamlit."""
    if fig is not None:
        st.pyplot(fig, use_container_width=True)

def add_message(role, content, plot=None, downloadable_data=None):
    """Add a message to the conversation, optionally with downloadable data."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        'role': role,
        'content': content,
        'timestamp': timestamp,
        'plot': plot,
        'downloadable_data': downloadable_data
    })

def generate_data_insights(df):
    """Generate interesting insights about the dataset."""
    # Use data exploration prompt for better guidance
    exploration_prompt = st.session_state.prompt_manager.get_data_exploration_prompt()
    
    insights = []
    
    # Basic stats
    insights.append(f"📊 **Dataset Overview**")
    insights.append(f"• Total records: {len(df):,}")
    insights.append(f"• Total columns: {len(df.columns)}")
    insights.append(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    insights.append(f"\n📈 **Column Types**")
    insights.append(f"• Numeric columns: {len(numeric_cols)}")
    insights.append(f"• Categorical columns: {len(categorical_cols)}")
    insights.append(f"• Datetime columns: {len(datetime_cols)}")
    
    # Missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        insights.append(f"\n⚠️ **Missing Values**")
        for col, missing in missing_data[missing_data > 0].items():
            percentage = (missing / len(df)) * 100
            insights.append(f"• {col}: {missing:,} ({percentage:.1f}%)")
    else:
        insights.append(f"\n✅ **No Missing Values**")
    
    # Interesting insights for numeric columns
    if numeric_cols:
        insights.append(f"\n🔍 **Numeric Insights**")
        for col in numeric_cols[:3]:  # Show first 3 numeric columns
            if df[col].dtype in ['int64', 'float64']:
                insights.append(f"• {col}:")
                insights.append(f"  - Min: {df[col].min():.2f}")
                insights.append(f"  - Max: {df[col].max():.2f}")
                insights.append(f"  - Mean: {df[col].mean():.2f}")
                insights.append(f"  - Median: {df[col].median():.2f}")
                # Outlier detection (IQR method)
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if not outliers.empty:
                    insights.append(f"  - Outliers: {len(outliers)} (values < {lower_bound:.2f} or > {upper_bound:.2f})")
                # Skewness and kurtosis
                try:
                    col_skew = skew(df[col].dropna())
                    col_kurt = kurtosis(df[col].dropna())
                    insights.append(f"  - Skewness: {col_skew:.2f}, Kurtosis: {col_kurt:.2f}")
                except Exception:
                    pass
    # Correlation matrix (top correlated pairs)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
        if not max_corr.empty:
            top_pair = max_corr.idxmax()
            insights.append(f"\n🔗 **Top Correlated Columns:** {top_pair[0]} & {top_pair[1]} (corr={max_corr.max():.2f})")
    # Unique/constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    all_unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if constant_cols:
        insights.append(f"\nℹ️ **Constant Columns:** {', '.join(constant_cols)} (only one unique value)")
    if all_unique_cols:
        insights.append(f"\n🆔 **All Unique Columns:** {', '.join(all_unique_cols)} (potential IDs)")
    # Text length/pattern analysis for categorical columns
    if categorical_cols:
        insights.append(f"\n✏️ **Text Length Analysis (Categorical Columns)**")
        for col in categorical_cols[:3]:
            lengths = df[col].dropna().astype(str).map(len)
            if not lengths.empty:
                insights.append(f"• {col}: min {lengths.min()}, max {lengths.max()}, avg {lengths.mean():.1f} chars")
    
    return "\n".join(insights)

def display_conversation():
    """Display the conversation history."""
    for message in st.session_state.messages:
        with st.container():
            if message['role'] == 'user':
                st.markdown(f"**You ({message['timestamp']}):**\n{message['content']}")
            else:
                content = message['content']
                st.markdown(f"**Assistant ({message['timestamp']}):**\n{content}")
                # Display plot if available
                if message.get('plot') is not None:
                    display_plot(message['plot'])

def main():
    """Main Streamlit app function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 SmartPandasAgent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Conversational Data Analysis with Natural Language</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            ["anthropic", "google", "openai"],
            index=0 if get_config_manager().llm_config.default_provider == "anthropic" else
            1 if get_config_manager().llm_config.default_provider == "google" else 2
        )
        
        # Model selection based on provider using config mappings
        available_models = list(get_config_manager().get_available_models(provider).keys())
        if available_models:
            # Get default model for the selected provider
            default_model = get_config_manager().get_model_name(provider, None)
            default_index = available_models.index(default_model) if default_model in available_models else 0
            model = st.selectbox("Model", available_models, index=default_index)
        else:
            st.error(f"No models available for provider: {provider}")
            return
        
        # Temperature
        temperature = st.slider("Temperature", 0.0, 1.0, get_config_manager().llm_config.default_temperature, 0.1)
        

        
        # CSV file upload
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        # Or use default file
        use_default = st.checkbox("Use default dataset", value=False)
        
        # Load data button
        if st.button("🚀 Load Data & Initialize Agent"):
            with st.spinner("Loading data and initializing agent..."):
                if use_default:
                    # Try to find default datasets
                    default_paths = [
                        "data/formated_data/professionals_in_pg.csv",
                        "data/raw/professionals_in_pg.csv",
                        "data.csv",
                        "dataset.csv",
                        "data/raw/data.csv"
                    ]
                    
                    csv_path = None
                    for path in default_paths:
                        if os.path.exists(path):
                            csv_path = path
                            break
                    
                    if csv_path:
                        st.info(f"🔄 Using default dataset: {csv_path}")
                    else:
                        st.error("No default dataset found. Please upload a CSV file.")
                        return
                    
                    # Load data with production-grade preprocessing (default folder)
                    st.info("🔄 Loading and preprocessing data...")
                    df, error = load_data(csv_path, is_uploaded_file=False)
                    if error:
                        st.error(f"Error loading data: {error}")
                        return
                    
                    # Use the same path for agent initialization
                    formatted_csv_path = csv_path
                    
                elif uploaded_file is not None:
                    # Create temporary directory for uploaded files
                    temp_dir = create_temp_directory()
                    
                    # Save uploaded file to temp directory with original name
                    original_csv_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(original_csv_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load data with production-grade preprocessing (temp folder)
                    st.info("🔄 Loading and preprocessing data...")
                    df, error = load_data(original_csv_path, is_uploaded_file=True)
                    if error:
                        st.error(f"Error loading data: {error}")
                        return
                    
                    # The formatted file is now the same as the original (overwritten during preprocessing)
                    # So we can use the original path for agent initialization
                    formatted_csv_path = original_csv_path
                    if not os.path.exists(formatted_csv_path):
                        st.error(f"Formatted file not found: {formatted_csv_path}")
                        return
                else:
                    st.error("Please upload a CSV file or use the default dataset.")
                    return
                
                # Initialize agent with the formatted file path
                agent, error = initialize_agent(formatted_csv_path, provider, model, temperature)
                if error:
                    st.error(f"Error initializing agent: {error}")
                    return
                
                # Store in session state
                st.session_state.df = df
                st.session_state.agent = agent
                st.session_state.csv_path = formatted_csv_path
                st.session_state.conversation_started = True
                
                # Clear previous messages
                st.session_state.messages = []
                
                st.success("✅ Data loaded, preprocessed, and agent initialized successfully!")
                st.rerun()
        
        # Data Summary Section
        if st.session_state.df is not None:
            st.header("📊 Data Summary")
            
            # Show preprocessing status
            st.subheader("🔄 Preprocessing Status")
            if st.session_state.temp_dir:
                st.success("✅ Data has been preprocessed (temporary session):")
                st.write("• Column names converted to lowercase")
                st.write("• String values converted to lowercase")
                st.write(f"• Formatted version saved to temporary directory")
                st.info("💡 Temporary files will be cleaned up when session ends")
            else:
                st.success("✅ Data has been preprocessed (persistent):")
                st.write("• Column names converted to lowercase")
                st.write("• String values converted to lowercase")
                st.write("• Formatted version saved to `data/formated_data/`")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📋 Sample Data", "🔍 Insights", "📊 Statistics"])
            
            with tab1:
                st.write(f"**Shape:** {st.session_state.df.shape}")
                st.write(f"**Columns:** {len(st.session_state.df.columns)}")
                st.write(f"**Memory Usage:** {st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Column types summary
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
                datetime_cols = st.session_state.df.select_dtypes(include=['datetime']).columns.tolist()
                
                st.write(f"**Numeric Columns:** {len(numeric_cols)}")
                st.write(f"**Categorical Columns:** {len(categorical_cols)}")
                st.write(f"**Datetime Columns:** {len(datetime_cols)}")
                
                # Missing values summary
                missing_data = st.session_state.df.isnull().sum()
                if missing_data.sum() > 0:
                    st.warning(f"⚠️ Missing values detected: {missing_data.sum()} total")
                else:
                    st.success("✅ No missing values")
            
            with tab2:
                st.subheader("First 10 Rows")
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                
                st.subheader("Last 5 Rows")
                st.dataframe(st.session_state.df.tail(5), use_container_width=True)
            
            with tab3:
                st.markdown(generate_data_insights(st.session_state.df))
            
            with tab4:
                if numeric_cols:
                    st.subheader("Numeric Statistics")
                    st.dataframe(st.session_state.df[numeric_cols].describe(), use_container_width=True)
                
                if categorical_cols:
                    st.subheader("Categorical Statistics")
                    for col in categorical_cols[:5]:  # Show first 5 categorical columns
                        st.write(f"**{col}:**")
                        value_counts = st.session_state.df[col].value_counts().head(10)
                        st.dataframe(value_counts.reset_index().rename(columns={'index': col, col: 'Count'}), use_container_width=True)
                        st.write("---")
        
        # LangSmith Status (Developer Mode)
        developer_mode = st.checkbox("🔧 Developer Mode", help="Show advanced debugging and monitoring features")
        st.session_state.developer_mode = developer_mode
        
        if developer_mode:
            render_langsmith_status()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            # Clean up temp directory when clearing conversation
            cleanup_temp_directory()
            st.rerun()
    
    # Main content area
    if not st.session_state.conversation_started:
        st.info("👈 Please configure the settings in the sidebar and load your data to start the conversation.")
        
        # Show example queries
        st.header("💡 Example Queries")
        st.markdown("""
        Once you load your data, you can ask questions like:
        - "How many records are there?"
        - "What is the average value by category?"
        - "Show me a bar chart of values by group"
        - "Which categories have the most items?"
        - "Create a histogram of numeric values"
        - "Top 5 highest values"
        - "Bottom 3 lowest values"
        - "Filter by specific criteria"
        """)
        
    else:
        # Data Summary in Main Area
        st.header("📊 Dataset Summary")
        
        # Create columns for summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(st.session_state.df):,}")
        
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        
        with col3:
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
            st.metric("Numeric Columns", len(numeric_cols))
        
        with col4:
            categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.metric("Categorical Columns", len(categorical_cols))
        
        # Quick insights
        st.subheader("🔍 Quick Insights")
        
        # Show some interesting facts
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.write("**📈 Numeric Insights:**")
            if numeric_cols:
                for col in numeric_cols[1:3]:  # Show first 2 numeric columns
                    if st.session_state.df[col].dtype in ['int64', 'float64']:
                        st.write(f"• **{col}**: {st.session_state.df[col].mean():.2f} (avg)")
        
        with insights_col2:
            st.write("**🏷️ Categorical Insights:**")
            if categorical_cols:
                for col in categorical_cols[:2]:  # Show first 2 categorical columns
                    top_value = st.session_state.df[col].value_counts().index[0]
                    top_count = st.session_state.df[col].value_counts().iloc[0]
                    st.write(f"• **{col}**: {top_value} ({top_count} records)")
        
        # Sample data preview
        st.subheader("📋 Sample Data")
        st.dataframe(st.session_state.df.head(5), use_container_width=True)
        
        # Subtle LangSmith indicator (only in developer mode)
        if st.session_state.get('developer_mode', False):
            langsmith_manager = get_langsmith_manager()
            if langsmith_manager.is_enabled:
                st.caption("🔍 LangSmith monitoring active")
        
        # LangSmith Observability Tab (Developer Mode)
        if st.session_state.get('developer_mode', False):
            st.subheader("🔍 LangSmith Observability")
            create_langsmith_tab()
        
        # Suggested questions based on data structure
        st.subheader("💡 Suggested Questions")
        
        # Show question parsing status
        st.info("🔍 **Question Parsing Active:** Your queries will be automatically corrected for typos and aligned with the dataset schema.")
        
        # Use visualization suggestion prompt for better guidance
        data_summary = f"Dataset with {len(st.session_state.df)} records, {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns"
        viz_suggestion_prompt = st.session_state.prompt_manager.get_visualization_suggestion_prompt(data_summary)
        
        # Generate questions based on column types
        suggested_questions = []
        
        if numeric_cols:
            for col in numeric_cols[1:3]:
                suggested_questions.append(f"What is the average {col}?")
                suggested_questions.append(f"Show me a histogram of {col}")
                suggested_questions.append(f"What is the distribution of {col}?")
        
        if categorical_cols:
            for col in categorical_cols[:2]:
                suggested_questions.append(f"How many records are in each {col}?")
                suggested_questions.append(f"Show me a bar chart of {col}")
                suggested_questions.append(f"What is the most common {col}?")
        
        if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
            suggested_questions.append(f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?")
        
        # Display suggested questions in columns
        if suggested_questions:
            cols = st.columns(2)
            for i, question in enumerate(suggested_questions[:6]):  # Show first 6 questions
                with cols[i % 2]:
                    if st.button(question, key=f"suggested_{i}", use_container_width=True):
                        process_query(question)
        
        st.markdown("---")
        
        # Display conversation
        st.header("💬 Conversation")
        
        # Display existing messages
        display_conversation()
        
        # Chat input
        st.markdown("---")
        
        # Query input
        query = st.text_input(
            "Ask a question about your data:",
                            placeholder="e.g., How many records are there?",
            key="query_input"
        )
        
        # Send button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🚀 Send", use_container_width=True):
                if query.strip():
                    process_query(query)
        with col2:
            if st.button("📊 Info", use_container_width=True):
                process_query("info")
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📈 Show Dataset Info", use_container_width=True):
                process_query("info")
        
        with col2:
            if st.button("🔄 Reset Filters", use_container_width=True):
                st.warning("Step-back context is not available in the new workflow-centric architecture. All context/entity memory is managed inside the workflow.")
        
        with col3:
            if st.button("📊 Show Columns", use_container_width=True):
                process_query("What columns are available in the dataset?")
        
        with col4:
            if st.button("📋 Sample Data", use_container_width=True):
                process_query("Show me the first 5 rows of data")
        
        with col5:
            if st.button("⬅️ Step Back Filter", use_container_width=True):
                st.warning("Step-back context is not available in the new workflow-centric architecture. All context/entity memory is managed inside the workflow.")

def process_query(query):
    """Process a user query with production-grade question parsing and advanced meta-question support."""
    if not st.session_state.agent:
        st.error("Agent not initialized. Please load data first.")
        return

    q_lower = query.lower().strip()
    config = get_config_manager()
    reset_commands = [cmd.lower() for cmd in config.context_config.reset_commands]
    # --- Advanced meta-question handling ---
    # Extract keyword for filtering if present
    question_filter = None
    answer_filter = None
    # e.g. "questions about salary" or "answers about age"
    q_about = re.search(r"questions? (about|containing|with) ([\w\s]+)", q_lower)
    a_about = re.search(r"answers? (about|containing|with) ([\w\s]+)", q_lower)
    qa_about = re.search(r"q&a pairs? (about|containing|with) ([\w\s]+)", q_lower)
    if q_about:
        question_filter = q_about.group(2).strip()
    if a_about:
        answer_filter = a_about.group(2).strip()
    if qa_about:
        qa_filter = qa_about.group(2).strip()
    else:
        qa_filter = None

    # Show all user questions
    if ("all question" in q_lower or "questions asked" in q_lower or "list my questions" in q_lower or "show my questions" in q_lower or q_about):
        questions = [
            m['content'] for m in st.session_state.messages if m['role'] == 'user'
        ]
        if question_filter:
            questions = [q for q in questions if question_filter in q.lower()]
        if questions:
            result = "### Questions you've asked so far:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            result = "No questions found matching your request."
        add_message('assistant', result)
        st.rerun()
        return

    # Show all assistant answers
    if ("all answer" in q_lower or "list answers" in q_lower or "show answers" in q_lower or a_about):
        answers = [
            m['content'] for m in st.session_state.messages if m['role'] == 'assistant'
        ]
        if answer_filter:
            answers = [a for a in answers if answer_filter in a.lower()]
        if answers:
            result = "### Answers so far:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
        else:
            result = "No answers found matching your request."
        add_message('assistant', result)
        st.rerun()
        return

    # Show Q&A pairs
    if ("q&a" in q_lower or "question answer" in q_lower or "qa pair" in q_lower or qa_about):
        pairs = []
        user_q = None
        for m in st.session_state.messages:
            if m['role'] == 'user':
                user_q = m['content']
            elif m['role'] == 'assistant' and user_q:
                pairs.append((user_q, m['content']))
                user_q = None
        if qa_filter:
            pairs = [p for p in pairs if qa_filter in p[0].lower() or qa_filter in p[1].lower()]
        if pairs:
            # Markdown table
            result = "| # | Question | Answer |\n|---|---------|--------|\n" + "\n".join(
                f"| {i+1} | {p[0].replace('|',' ')} | {p[1].replace('|',' ')} |" for i, p in enumerate(pairs)
            )
        else:
            result = "No Q&A pairs found matching your request."
        add_message('assistant', result)
        st.rerun()
        return

    # --- Handle reset command before normal query flow ---
    if q_lower in reset_commands:
        result, plot = st.session_state.agent.query(query)
        st.session_state.df = st.session_state.agent.df
        add_message('assistant', result, plot)
        st.rerun()
        return

    # --- Normal query flow below ---
    add_message('user', query)
    with st.spinner("🤖 Processing your question..."):
        try:
            question_parser_provider = st.session_state.agent.provider
            if not config.validate_provider_config(question_parser_provider):
                st.error(f"Invalid provider configuration for question parsing: {question_parser_provider}")
                return
            with st.spinner("🔍 Parsing and correcting your question..."):
                rewritten_query = rewrite_user_question(
                    question=query,
                    df=st.session_state.df,
                    provider=question_parser_provider,
                    model_name=st.session_state.agent.model
                )
            if rewritten_query.lower() != query.lower():
                st.info(f"🔧 **Query corrected:** '{query}' → '{rewritten_query}'")
            result, plot = st.session_state.agent.query(rewritten_query)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
                add_message('assistant', "See table above.", plot)
            else:
                add_message('assistant', result, plot)
            st.rerun()
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            add_message('assistant', error_msg)
            st.rerun()

def cleanup_temp_directory():
    """Clean up temporary directory if it exists."""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            logger.info(f"✅ Temporary directory cleaned up: {st.session_state.temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Warning: Could not clean up temp directory: {e}")

def create_temp_directory():
    """Create a temporary directory for uploaded files."""
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="chat_csv_")
        logger.info(f"📁 Created temporary directory: {st.session_state.temp_dir}")
    return st.session_state.temp_dir

if __name__ == "__main__":
    # Register cleanup on app exit
    atexit.register(cleanup_temp_directory)
    main() 