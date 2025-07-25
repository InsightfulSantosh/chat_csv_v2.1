# SmartPandasAgent - Production-Ready Modular System

A production-ready, modular pandas DataFrame assistant that uses natural language to query and visualize data with advanced fuzzy matching, context awareness, and multi-provider LLM support.

## 🏗️ Clean Modular, Workflow-Centric Architecture

The codebase now features a **workflow-centric** architecture powered by LangGraph, with all context/entity memory filtering and DataFrame management handled inside the workflow nodes. The agent is now a thin orchestrator, and all state transitions are explicit and testable.

```
chat_csv_v1/
├── .env                    # Environment configuration (create this)
├── .gitignore             # Git ignore rules
├── main.py                # CLI interface
├── app.py                 # Streamlit web application
├── requirements.txt       # Dependencies
├── smart_pandas_agent.py  # Main orchestrator class (now minimal glue code)
├── README.md              # This documentation
├── QUICK_START.md         # Quick start guide
├── venv/                  # Virtual environment
├── agents/                # Agent modules
│   ├── __init__.py
│   ├── conditional_langgraph_agent.py # LangGraph conditional workflow (all context logic here)
│   └── llm_agent.py       # LLM agent management
└── utils/                 # Utility modules
    ├── __init__.py
    ├── logger.py          # Logging configuration
    ├── fuzzy_matcher.py   # Fuzzy matching for columns/values
    ├── code_fixer.py      # Pandas syntax corrections
    ├── context_filter.py  # Context-aware filtering (used by workflow nodes)
    ├── plot_manager.py    # Plotting functionality
    ├── query_executor.py  # Query execution and retry logic
    ├── tool.py            # Node logic for LangGraph
    ├── prompts.py         # Template-driven prompt management
    ├── question_parser.py # Question parsing and rewriting
    ├── data_loader.py     # CSV loading and preprocessing
    ├── llm_factory.py     # Multi-provider LLM management
    ├── config_manager.py  # Centralized configuration management
    ├── provider_registry.py # LLM provider registry
    ├── langsmith_config.py # LangSmith observability
    ├── langsmith_dashboard.py # LangSmith dashboard
    ├── query_analyzer.py  # Query analysis and routing
    └── base_dataframe_manager.py # Base DataFrame management
```

## 🚀 Key Features

- **🎯 Multi-Provider LLM Support**: Seamlessly switch between Anthropic (Claude), Google (Gemini), and OpenAI (GPT) models
- **🧠 Intelligent Fuzzy Matching**: Advanced column and value matching with configurable thresholds
- **🔄 Context Awareness**: Maintains conversation context and entity memory **inside the workflow**
- **📊 Advanced Plotting**: Support for multiple plot types with automatic styling and customization
- **🛡️ Robust Error Handling**: Comprehensive retry mechanisms and graceful error recovery
- **🔒 Security First**: Safe code execution with forbidden pattern detection and validation
- **⚙️ Centralized Configuration**: Easy configuration management with environment variables
- **🔍 Smart Question Parsing**: Intelligent question rewriting and schema alignment
- **📁 Automated Data Loading**: CSV preprocessing with lowercase conversion and column renaming
- **🌐 Web Interface**: Beautiful Streamlit web app with conversational UI
- **🏗️ Modular Design**: Clean separation of concerns with dedicated, reusable modules
- **📈 Data Type Awareness**: Correct handling of categorical vs numeric data for queries
- **🔧 Template-Driven Prompts**: Production-ready prompt management system
- **📊 LangSmith Integration**: Optional observability and debugging support

## 🧩 How the Workflow Works (New Architecture)

- **All context/entity memory filtering and DataFrame management is now handled inside LangGraph nodes.**
- The workflow state includes:
  - `full_df`: The original DataFrame
  - `context_filter_state`: Serializable context/entity memory state
  - `filtered_df`: The DataFrame after filtering (updated by a node)
  - `query`: The user’s query
  - Other metadata (messages, tool results, etc.)
- The workflow nodes:
  1. `apply_context_filter`: Applies context/entity memory filtering to `full_df` using the query and context state
  2. `analyze_query`: Analyzes the query for intent and routing
  3. `data_query` / `visualization`: Executes the query or creates a plot using `filtered_df`
  4. `format_response`: Formats the final response
- **No context stack or context filter logic in the agent.**
- **Minimal glue code:** The agent simply passes the user’s question to the workflow.

## 📋 Requirements

- **Python**: 3.8+ (tested with Python 3.12.4)
- **API Keys**: For your chosen LLM provider:
  - **Anthropic**: `ANTHROPIC_API_KEY` (Claude models)
  - **Google**: `GOOGLE_API_KEY` (Gemini models)
  - **OpenAI**: `OPENAI_API_KEY` (GPT models)
- **Dependencies**: See `requirements.txt` for complete list

## 🔧 Installation & Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd chat_csv_v1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Choose your LLM provider (anthropic, google, openai)
DEFAULT_PROVIDER=anthropic

# API Keys (only the one for your chosen provider is required)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Model configuration (optional - uses provider defaults)
DEFAULT_MODEL=claude-3-5-sonnet-20240620
DEFAULT_TEMPERATURE=0
MAX_RETRIES=3
RETRY_DELAY=0.5

# Logging
LOG_LEVEL=INFO

# Optional: LangSmith for observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=smart-pandas-agent
```

### 3. Get API Keys

- **Anthropic**: [Console](https://console.anthropic.com/) → API Keys
- **Google**: [MakerSuite](https://makersuite.google.com/app/apikey) → API Keys
- **OpenAI**: [Platform](https://platform.openai.com/api-keys) → API Keys
- **LangSmith (Optional)**: [Smith](https://smith.langchain.com/) → API Keys

## 🎯 Quick Start

### Option 1: Web Interface (Recommended)

For the best user experience with beautiful visualizations:

```bash
# Start the web interface
streamlit run app.py

# Features:
# - 📊 Interactive data visualization with real-time plots
# - 💬 Conversational chat interface with context memory
# - ⚙️ Easy provider/model switching
# - 📁 Drag-and-drop file upload capability
# - 🎨 Beautiful UI with streaming responses
# - 🔍 Advanced question parsing and correction
# - 📈 Data insights and suggested questions
```

### Option 2: Command Line Interface

For quick command-line usage:

```bash
# Start interactive session
python main.py

# The CLI will guide you through:
# 1. Selecting your CSV file
# 2. Choosing LLM provider (if multiple keys configured)
# 3. Interactive querying with natural language
# 4. Automatic question parsing and correction
```

### Basic Python Usage

```python
from smart_pandas_agent import SmartPandasAgent

# Initialize with default provider
agent = SmartPandasAgent("path/to/your/data.csv")

# Or specify provider explicitly
agent = SmartPandasAgent("path/to/your/data.csv", provider="anthropic")
agent = SmartPandasAgent("path/to/your/data.csv", provider="google")
agent = SmartPandasAgent("path/to/your/data.csv", provider="openai")

# Query your data
result, plot = agent.query("Show me the average salary by department")

# Save plot if generated
if plot:
    agent.save_plot("output.png")
```

## 🔄 Multi-Provider LLM Usage

### Automatic Provider Selection

```python
from smart_pandas_agent import SmartPandasAgent

# Uses DEFAULT_PROVIDER from .env
agent = SmartPandasAgent("data.csv")

# Or override at runtime
agent = SmartPandasAgent("data.csv", provider="google", model="gemini-1.5-pro")
```

### Available Models

The system supports multiple models per provider with automatic model mapping:

```python
from utils.llm_factory import get_available_models

# Get available models for each provider
anthropic_models = get_available_models("anthropic")
google_models = get_available_models("google")
openai_models = get_available_models("openai")

print(f"Anthropic: {list(anthropic_models.keys())}")
print(f"Google: {list(google_models.keys())}")
print(f"OpenAI: {list(openai_models.keys())}")
```

**Available Models:**
- **Anthropic**: `claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`, `gemini-2.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Provider-Specific Features

- **Anthropic (Claude)**: Best for complex reasoning and analysis
- **Google (Gemini)**: Excellent for data visualization and quick responses
- **OpenAI (GPT)**: Great balance of speed and accuracy

## 🔍 Advanced Features

### Intelligent Question Parsing

```python
from utils.question_parser import rewrite_user_question
import pandas as pd

df = pd.read_csv("your_data.csv")

# Fix typos and align with schema
original = "Show me all employes in the eng department"
rewritten = rewrite_user_question(original, df, provider="google")
print(f"Original: {original}")
print(f"Rewritten: {rewritten}")
```

### Data Loading & Preprocessing

```python
from utils.data_loader import load_csv_with_preprocessing

# Load with automatic preprocessing
df = load_csv_with_preprocessing(
    input_path="your_data.csv",
    lowercase=True,
    rename_columns={"old_name": "new_name"},
    save_formatted=True
)
```

### Configuration Management

```python
from utils.config_manager import get_config_manager

# Access any configuration value
config = get_config_manager()
print(f"Default model: {config.llm_config.default_model}")
print(f"Max retries: {config.llm_config.max_retries}")
print(f"Security level: {config.security_config.security_level}")
```

## 📊 Supported Plot Types

- **Bar Plots**: Grouped, stacked, and count plots
- **Line Plots**: Time series and trend analysis
- **Scatter Plots**: Correlation and relationship analysis
- **Histograms**: Distribution analysis
- **Box Plots**: Statistical summaries
- **Pie Charts**: Proportional data visualization
- **Heatmaps**: Correlation matrices and complex relationships
- **Count Plots**: Categorical data frequency analysis

## 🔧 Query Examples

### Data Analysis Queries
```python
# Basic statistics
"Show me the average salary by department"
"What is the correlation between age and salary?"
"Count the number of employees in each department"

# Filtering and grouping
"Show me employees older than 30"
"Find departments with average salary above 70000"
"List employees in the Engineering department"

# Ranking and top/bottom queries
"Top 5 highest paid employees"
"Bottom 3 departments by employee count"
"Most common job titles"
```

### Visualization Queries
```python
# Plot requests
"Create a bar chart of salaries by department"
"Show me a histogram of employee ages"
"Plot the correlation between experience and salary"
"Generate a pie chart of department distribution"
```

## 🛡️ Security Features

### Code Execution Safety
- **Forbidden Pattern Detection**: Blocks dangerous code patterns
- **Environment Isolation**: Safe execution environment
- **Input Validation**: Validates all user inputs
- **Error Handling**: Graceful error recovery

### Security Levels
- **Production**: Most restrictive, blocks all potentially dangerous operations
- **Staging**: Moderate restrictions for testing
- **Development**: Relaxed restrictions for development

## 📈 Performance Optimizations

### Recent Improvements
- **Perfect Import Hygiene**: All imports properly organized and optimized
- **Template-Driven Prompts**: Efficient prompt management system
- **Data Type Awareness**: Correct handling of categorical vs numeric data
- **Optional Dependencies**: Graceful handling of missing packages
- **Faster Module Loading**: Optimized import structure

### Best Practices
1. **Use the Web Interface**: Best performance and user experience
2. **Choose Appropriate Models**: Balance speed vs accuracy
3. **Optimize Queries**: Be specific and clear in your questions
4. **Use Context**: Leverage conversation memory for complex queries

## 🚨 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check your .env file
cat .env

# Verify the key is loaded
python -c "from utils.config_manager import get_config_manager; config = get_config_manager(); print('Provider:', config.llm_config.default_provider)"
```

#### 2. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 3. Configuration Issues
```python
# Test configuration loading
python -c "from utils.config_manager import get_config_manager; config = get_config_manager(); print('Config loaded successfully')"
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in .env
LOG_LEVEL=DEBUG
```

## 🔍 What's New in This Version

### ✨ Recent Improvements
- **🎯 Perfect Import Hygiene**: All imports properly organized and optimized
- **🧠 Enhanced Prompt System**: Template-driven prompts with data type awareness
- **🔧 Fixed Query Issues**: Correct handling of categorical vs numeric data
- **📊 Better Error Handling**: Graceful handling of missing optional dependencies
- **⚡ Performance Optimizations**: Faster module loading and execution
- **🎨 Improved UI**: Enhanced Streamlit interface with better UX
- **🔍 Advanced Question Parsing**: Better typo correction and schema alignment

### 🚀 Key Features
- **Multi-Provider LLM Support**: Anthropic, Google, OpenAI
- **Intelligent Question Parsing**: Automatic typo correction and schema alignment
- **Advanced Plotting**: Multiple plot types with automatic styling
- **Context Awareness**: Maintains conversation context across queries
- **Robust Error Handling**: Comprehensive retry mechanisms
- **Security First**: Safe code execution with pattern detection
- **Modular Architecture**: Clean, maintainable codebase

## 📚 Documentation

- **Quick Start**: See `QUICK_START.md` for getting started in 5 minutes
- **Configuration**: Review environment variables and settings
- **Examples**: Check the code examples in this guide
- **Troubleshooting**: See the troubleshooting section above

## 🤝 Contributing

This is a production-ready system with clean, modular architecture. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Ready to Start!

Your SmartPandasAgent is now ready for production use! Start with the web interface for the best experience:

```bash
streamlit run app.py
```

Happy data analysis! 🚀📊 