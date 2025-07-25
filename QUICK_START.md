# 🚀 Quick Start Guide - Chat CSV v2.1

Get up and running with Chat CSV v2.1 in minutes! This guide will help you set up and start using the production-ready system with your own data.

## ⚡ Super Quick Setup (5 minutes)

### 1. Environment Setup
```bash
# Navigate to your project directory
cd chat_csv_v2.1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```bash
# Choose your LLM provider
DEFAULT_PROVIDER=anthropic

# Add your API key (only one required)
ANTHROPIC_API_KEY=your_actual_anthropic_api_key_here

# Optional: Add other providers for flexibility
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith for observability
LANGCHAIN_TRACING_V2.1=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=chat-csv-v2.1
```

### 3. Get Your API Key
- **Anthropic (Claude)**: [Get API Key](https://console.anthropic.com/)
- **Google (Gemini)**: [Get API Key](https://makersuite.google.com/app/apikey)
- **OpenAI (GPT)**: [Get API Key](https://platform.openai.com/api-keys)
- **LangSmith (Optional)**: [Get API Key](https://smith.langchain.com/)

## 🎯 Start Using Chat CSV v2.1

### Option 1: Web Interface (Recommended)
```bash
# Start the beautiful Streamlit web interface
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

### Option 2: Interactive CLI
```bash
# Start the interactive command-line interface
python main.py

# The CLI will guide you through:
# 1. Selecting your CSV file
# 2. Choosing LLM provider (if multiple configured)
# 3. Interactive querying with natural language
# 4. Automatic question parsing and correction
```

## 📊 Example with Sample Data

### Create Sample Dataset
```python
import pandas as pd

# Create sample employee data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
    'Salary': [75000, 65000, 70000, 80000, 60000],
    'Experience_Years': [2, 5, 8, 3, 4]
}
df = pd.DataFrame(data)
df.to_csv('sample_employees.csv', index=False)
```

### Query Your Data
You can use the web interface or the CLI to query the sample data.

## 🔄 Multi-Provider LLM Usage

### Switch Between Providers
You can switch between providers in the web interface or by setting the `DEFAULT_PROVIDER` in your `.env` file.

### Provider-Specific Features
- **Anthropic (Claude)**: Best for complex analysis and reasoning
- **Google (Gemini)**: Excellent for data visualization and quick responses
- **OpenAI (GPT)**: Great balance of speed and accuracy

## 🛠️ Advanced Features

### Intelligent Question Parsing
The system uses an LLM to parse and understand your queries, including complex, multi-part questions.

### Data Loading & Preprocessing
```python
from utils.data_loader import load_csv_with_preprocessing

# Load with automatic preprocessing
df = load_csv_with_preprocessing(
    input_path="your_data.csv",
    lowercase=True,  # Convert column names and values to lowercase
    rename_columns={"old_name": "new_name"},  # Custom column renaming
    save_formatted=True  # Save the processed version
)
```

### Configuration Management
```python
from utils.config_manager import get_config_manager

config = get_config_manager()
print(f"Default provider: {config.llm_config.default_provider}")
print(f"Default model: {config.llm_config.default_model}")
print(f"Max retries: {config.llm_config.max_retries}")
```

## 🚨 Troubleshooting

### Common Issues & Solutions

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

## 🎯 Environment Management

### Activate Environment
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Deactivate Environment
```bash
deactivate
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## 📈 Performance Tips

1. **Choose the Right Model**:
   - **Speed**: Use smaller models (haiku, flash, mini)
   - **Accuracy**: Use larger models (opus, pro, turbo)

2. **Optimize Your Queries**:
   - Be specific: "Show me sales by region for Q1 2024"
   - Avoid vague: "Tell me about the data"

3. **Use the Web Interface**:
   - Better visualization capabilities
   - Context memory across conversations
   - Real-time streaming responses

## 🔍 What's New in This Version

### ✨ Recent Improvements
- **🧠 LLM-Powered Query Analysis**: The system now uses an LLM to analyze and understand queries, including complex, multi-part questions.
- **🔧 Improved Error Handling**: Better error handling for a smoother user experience.
- **⚡ Performance Optimizations**: Faster module loading and execution.
- **🎨 Improved UI**: Enhanced Streamlit interface with better UX.

### 🚀 Key Features
- **Multi-Provider LLM Support**: Anthropic, Google, OpenAI
- **Intelligent Question Parsing**: Automatic typo correction and schema alignment
- **Advanced Plotting**: Multiple plot types with automatic styling
- **Context Awareness**: Maintains conversation context across queries
- **Robust Error Handling**: Comprehensive retry mechanisms
- **Security First**: Safe code execution with pattern detection
- **Modular Architecture**: Clean, maintainable codebase

## 🎉 Ready to Start!

Your Chat CSV v2.1 is now ready for use! Start with the web interface for the best experience:

```bash
streamlit run app.py
```

Happy data analysis! 🚀
