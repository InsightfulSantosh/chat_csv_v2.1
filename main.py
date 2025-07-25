#!/usr/bin/env python3
"""
Main CLI interface for SmartPandasAgent with question parsing.
Provides an interactive command-line interface for querying CSV data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from main_orchestrator import SmartPandasAgent
from utils.question_parser import rewrite_user_question
from utils.data_loader import validate_csv_path
from utils.config_manager import get_config_manager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CliManager:
    """Manages the command-line interface for the SmartPandasAgent."""
    
    def __init__(self):
        self.agent = None
        self.df = None
        self.config = get_config_manager()

    def _initialize_agent(self, csv_path: str):
        """Initializes the SmartPandasAgent."""
        try:
            logger.info("🤖 Initializing SmartPandasAgent...")
            self.agent = SmartPandasAgent(
                csv_path=csv_path,
                provider=self.config.llm_config.default_provider,
                model=self.config.llm_config.default_model,
                temperature=self.config.llm_config.default_temperature
            )
            self.df = self.agent.df
            logger.info("✅ SmartPandasAgent initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing agent: {e}")
            return False

    def _get_csv_path(self) -> str:
        """Prompts the user for a CSV file path or uses a default."""
        csv_path = input("📁 Enter CSV file path (or press Enter for default): ").strip()
        if not csv_path:
            default_paths = [
                "data/raw/professionals_in_pg.csv",
                "data.csv",
                "dataset.csv",
                "data/raw/data.csv"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    return path
            logger.error("❌ No default CSV file found. Please provide a path.")
            return None
        return csv_path

    def _process_query(self, query: str):
        """Processes a user query."""
        try:
            logger.info(f"🔍 Processing: {query}")
            question_parser_provider = "google"
            
            logger.info("🔄 Rewriting question...")
            rewritten_query = rewrite_user_question(query, self.df, provider=question_parser_provider)
            logger.info(f"🔁 Rewritten: {rewritten_query}")

            logger.info("🤖 Querying agent...")
            result, fig = self.agent.query(rewritten_query)

            print("\n📊 Answer:")
            print("-" * 40)
            print(result)
            print("-" * 40)

            if fig:
                logger.info("📈 Displaying plot...")
                plt.show()
            else:
                logger.info("📈 No plot generated for this query.")

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            print("💡 Try rephrasing your question or type 'help' for assistance.")

    def run(self):
        """Runs the main CLI loop."""
        print("🚀 SmartPandasAgent CLI")
        print("=" * 50)

        provider = self.config.llm_config.default_provider
        print(f"🤖 Using LLM provider: {provider.upper()}")

        if not self.config.validate_provider_config(provider):
            logger.error(f"❌ Error: {provider.upper()} API key not configured!")
            print(f"   Please set {provider.upper()}_API_KEY in your .env file.")
            return

        csv_path = self._get_csv_path()
        if not csv_path or not validate_csv_path(csv_path):
            logger.error(f"❌ Invalid or inaccessible CSV file: {csv_path}")
            return

        if not self._initialize_agent(csv_path):
            return

        print("\n💡 Available commands:")
        print("   - Type your question in natural language")
        print("   - Type 'info' to see dataset information")
        print("   - Type 'reset' to clear filters and memory")
        print("   - Type 'exit' or 'quit' to exit")
        print("   - Type 'help' to show this help again")
        print()

        while True:
            try:
                query = input("💬 Your question (type 'exit' to quit): ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    print("👋 Exiting...")
                    break
                elif query.lower() == 'help':
                    print("\n💡 Available commands:")
                    print("   - Type your question in natural language")
                    print("   - Type 'info' to see dataset information")
                    print("   - Type 'reset' to clear filters and memory")
                    print("   - Type 'exit' or 'quit' to exit")
                    print("   - Type 'help' to show this help again")
                elif query.lower() == 'info':
                    print("\n📊 Dataset Information:")
                    print(f"   Shape: {self.df.shape}")
                    print(f"   Columns: {list(self.df.columns)}")
                    print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                elif query.lower() == 'reset':
                    print("Reset filters is not available in the new workflow-centric architecture. All context/entity memory is managed inside the workflow.")
                else:
                    self._process_query(query)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Interrupted by user. Exiting...")
                break

def main():
    """Main entry point for the CLI application."""
    cli = CliManager()
    cli.run()

if __name__ == "__main__":
    main()
