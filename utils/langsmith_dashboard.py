"""
LangSmith Dashboard utilities for Streamlit app.

This module provides Streamlit components for viewing LangSmith runs,
metadata, and analytics within the SmartPandasAgent app.
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from utils.langsmith_config import get_langsmith_manager
from utils.logger import setup_logger

logger = setup_logger(__name__)

def render_langsmith_status():
    """Render LangSmith status in the sidebar."""
    langsmith_manager = get_langsmith_manager()
    
    st.sidebar.subheader("🔍 LangSmith Observability")
    
    if not langsmith_manager.is_enabled:
        st.sidebar.info("📊 LangSmith: Disabled")
        with st.sidebar.expander("🔧 Enable LangSmith"):
            st.code("""
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=smart-pandas-agent
            """)
        return False
    
    st.sidebar.success("✅ LangSmith: Enabled")
    
    # Show project info
    project_name = os.getenv("LANGCHAIN_PROJECT", "smart-pandas-agent")
    st.sidebar.caption(f"Project: {project_name}")
    
    # Show LangSmith URL
    langsmith_url = f"https://smith.langchain.com/project/{project_name}"
    st.sidebar.markdown(f"[🔗 Open Dashboard]({langsmith_url})")
    
    return True

def render_recent_runs(limit: int = 10):
    """Render recent LangSmith runs."""
    langsmith_manager = get_langsmith_manager()
    
    if not langsmith_manager.is_enabled:
        st.warning("LangSmith is not enabled")
        return
    
    st.subheader("📊 Recent Runs")
    
    try:
        runs = langsmith_manager.list_recent_runs(limit=limit)
        
        if not runs:
            st.info("No recent runs found")
            return
        
        # Convert runs to DataFrame for display
        run_data = []
        for run in runs:
            run_data.append({
                "Run ID": str(run.id),  # Convert UUID to string
                "Name": run.name or "Unnamed",
                "Status": run.status,
                "Start Time": run.start_time.isoformat() if run.start_time else "N/A",
                "End Time": run.end_time.isoformat() if run.end_time else "N/A",
                "Duration": str(run.end_time - run.start_time) if run.start_time and run.end_time else "N/A",
                "Provider": run.extra.get("provider", "Unknown") if run.extra else "Unknown",
                "Model": run.extra.get("model", "Unknown") if run.extra else "Unknown"
            })
        
        df = pd.DataFrame(run_data)
        st.dataframe(df, use_container_width=True)
        
        # Show run details on selection
        if st.button("📋 Show Run Details"):
            selected_run_id = st.selectbox(
                "Select a run to view details:",
                [run["Run ID"] for run in run_data]
            )
            
            if selected_run_id:
                run_details = langsmith_manager.get_run_details(selected_run_id)
                if run_details:
                    st.json(run_details.dict())
                else:
                    st.error("Failed to load run details")
    
    except Exception as e:
        st.error(f"Failed to load recent runs: {e}")

def render_analytics():
    """Render LangSmith analytics."""
    langsmith_manager = get_langsmith_manager()
    
    if not langsmith_manager.is_enabled:
        st.warning("LangSmith is not enabled")
        return
    
    st.subheader("📈 Analytics")
    
    try:
        runs = langsmith_manager.list_recent_runs(limit=100)
        
        if not runs:
            st.info("No runs available for analytics")
            return
        
        # Basic analytics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_runs = len(runs)
            st.metric("Total Runs", total_runs)
        
        with col2:
            successful_runs = len([r for r in runs if r.status == "completed"])
            st.metric("Successful Runs", successful_runs)
        
        with col3:
            failed_runs = len([r for r in runs if r.status == "failed"])
            st.metric("Failed Runs", failed_runs)
        
        # Provider distribution
        provider_counts = {}
        for run in runs:
            provider = run.extra.get("provider", "Unknown") if run.extra else "Unknown"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        if provider_counts:
            st.subheader("Provider Distribution")
            provider_df = pd.DataFrame([
                {"Provider": provider, "Count": count}
                for provider, count in provider_counts.items()
            ])
            st.bar_chart(provider_df.set_index("Provider"))
        
        # Average duration
        durations = []
        for run in runs:
            if run.start_time and run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
                durations.append(duration)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            st.metric("Average Duration", f"{avg_duration:.2f}s")
    
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

def render_run_metadata(run_id: str):
    """Render metadata for a specific run."""
    langsmith_manager = get_langsmith_manager()
    
    if not langsmith_manager.is_enabled:
        st.warning("LangSmith is not enabled")
        return
    
    run_details = langsmith_manager.get_run_details(run_id)
    
    if not run_details:
        st.error("Failed to load run details")
        return
    
    st.subheader(f"📋 Run Details: {run_id}")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Status:**", run_details.status)
        st.write("**Name:**", run_details.name or "Unnamed")
        st.write("**Start Time:**", run_details.start_time.isoformat() if run_details.start_time else "N/A")
    
    with col2:
        st.write("**End Time:**", run_details.end_time.isoformat() if run_details.end_time else "N/A")
        if run_details.start_time and run_details.end_time:
            duration = (run_details.end_time - run_details.start_time).total_seconds()
            st.write("**Duration:**", f"{duration:.2f}s")
    
    # Metadata
    if run_details.extra:
        st.subheader("📊 Metadata")
        st.json(run_details.extra)
    
    # Input/Output
    if run_details.inputs:
        st.subheader("📥 Inputs")
        st.json(run_details.inputs)
    
    if run_details.outputs:
        st.subheader("📤 Outputs")
        st.json(run_details.outputs)
    
    # Error info
    if run_details.error:
        st.subheader("❌ Error")
        st.error(run_details.error)

def create_langsmith_tab():
    """Create a complete LangSmith tab for the app."""
    st.header("🔍 LangSmith Observability")
    
    # Check if LangSmith is enabled
    langsmith_manager = get_langsmith_manager()
    
    if not langsmith_manager.is_enabled:
        st.warning("⚠️ LangSmith is not enabled")
        st.info("""
        To enable LangSmith observability, set these environment variables:
        
        ```bash
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=your_langsmith_api_key
        LANGCHAIN_PROJECT=smart-pandas-agent
        ```
        
        You can get your API key from [LangSmith](https://smith.langchain.com/).
        """)
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Recent Runs", "📈 Analytics", "🔍 Run Details"])
    
    with tab1:
        render_recent_runs()
    
    with tab2:
        render_analytics()
    
    with tab3:
        st.subheader("🔍 View Specific Run")
        run_id = st.text_input("Enter Run ID:")
        if run_id:
            render_run_metadata(run_id)
        else:
            st.info("Enter a Run ID to view detailed information") 