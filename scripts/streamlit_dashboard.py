#!/usr/bin/env python3
"""
Streamlit Dashboard for Multi-Agent Data Analysis Framework
Provides a web interface to interact with the orchestrator agent for data analysis
"""

import streamlit as st
import httpx
import json
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import time
from common_utils.output_manager import output_manager

# Configure page
st.set_page_config(
    page_title="Multi-Agent Data Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
ORCHESTRATOR_URL = "http://localhost:10000"
DATA_FOLDER = Path("data")

class OrchestratorClient:
    """Client for communicating with the orchestrator agent"""
    
    def __init__(self, base_url: str = ORCHESTRATOR_URL):
        self.base_url = base_url
        
    async def health_check(self) -> bool:
        """Check if orchestrator is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get orchestrator capabilities"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/capabilities")
                return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}
    
    async def orchestrate_pipeline(self, file_path: str, analysis_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a full analysis pipeline"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "orchestrate_pipeline",
                "params": {
                    "file_path": file_path,
                    "pipeline_config": analysis_config or {}
                },
                "id": 1
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout
                response = await client.post(self.base_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        return {"status": "error", "error": result["error"]}
                    return result.get("result", {})
                else:
                    return {"status": "error", "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            return {"status": "error", "error": str(e)}

def get_available_datasets() -> List[str]:
    """Get list of available datasets from the data folder"""
    if not DATA_FOLDER.exists():
        return []
    
    datasets = []
    for file_path in DATA_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.json', '.tdsx']:
            datasets.append(file_path.name)
    
    return sorted(datasets)

def get_file_info(filename: str) -> Dict[str, Any]:
    """Get file information"""
    file_path = DATA_FOLDER / filename
    if not file_path.exists():
        return {}
    
    stat = file_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    
    return {
        "size_mb": round(size_mb, 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": file_path.suffix.lower()
    }

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def format_pipeline_results(results: Dict[str, Any]) -> None:
    """Format and display pipeline results"""
    if not results:
        st.error("No results to display")
        return
    
    if results.get("status") == "error":
        st.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
        return
    
    # Display pipeline summary
    st.success("✅ Pipeline completed successfully!")
    
    # Show pipeline metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "pipeline_id" in results:
            st.metric("Pipeline ID", results["pipeline_id"][:8] + "...")
    
    with col2:
        if "started_at" in results:
            st.metric("Started At", results["started_at"])
    
    with col3:
        if "completed_at" in results:
            st.metric("Completed At", results["completed_at"])
    
    # Show stages
    if "stages" in results:
        st.subheader("🔄 Pipeline Stages")
        
        stages = results["stages"]
        for stage_name, stage_data in stages.items():
            with st.expander(f"📋 {stage_name.replace('_', ' ').title()}", expanded=False):
                
                # Status indicator
                status = stage_data.get("status", "unknown")
                if status == "completed":
                    st.success(f"Status: {status}")
                elif status == "skipped":
                    st.warning(f"Status: {status}")
                else:
                    st.info(f"Status: {status}")
                
                # Display stage-specific data
                for key, value in stage_data.items():
                    if key != "status":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Show final report URL if available
    if "final_report_url" in results:
        st.subheader("📊 Final Report")
        st.markdown(f"[View Full Report]({results['final_report_url']})")
    
    # Show analysis summary if available
    if "summary" in results:
        st.subheader("📈 Analysis Summary")
        summary = results["summary"]
        if isinstance(summary, dict):
            for key, value in summary.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.write(summary)

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🤖 Multi-Agent Data Analysis Dashboard")
    st.markdown("""
    Welcome to the Multi-Agent Data Analysis Framework! Select a dataset and describe what analysis you'd like to perform.
    The orchestrator will coordinate multiple specialized agents to process your data.
    """)
    
    # Initialize client
    client = OrchestratorClient()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Health check
        st.subheader("🏥 System Health")
        if st.button("Check Health"):
            with st.spinner("Checking orchestrator health..."):
                is_healthy = run_async(client.health_check())
                if is_healthy:
                    st.success("✅ Orchestrator is healthy")
                    # Get capabilities
                    capabilities = run_async(client.get_capabilities())
                    if capabilities:
                        st.json(capabilities)
                else:
                    st.error("❌ Orchestrator is not responding")
                    st.warning("Make sure the orchestrator agent is running on port 10000")
        
        # Analysis options
        st.subheader("🔧 Analysis Options")
        include_root_cause = st.checkbox("Include Root Cause Analysis", value=True, 
                                       help="Use the Why-Bot for root cause analysis")
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["quick", "standard", "comprehensive"],
            index=2,
            help="Level of analysis detail"
        )
        
        output_format = st.selectbox(
            "Output Format",
            ["html", "pdf", "markdown"],
            index=0,
            help="Format for the final report"
        )
        
        # Previous Sessions section
        st.subheader("📚 Previous Sessions")
        if st.button("🔄 Refresh Sessions"):
            st.rerun()
            
        # Get and display previous sessions
        previous_sessions = output_manager.list_all_sessions()
        
        if previous_sessions:
            # Show only the most recent 5 sessions in sidebar
            for session in previous_sessions[:5]:
                with st.expander(f"📊 {session['dataset_name'][:15]}...", expanded=False):
                    st.write(f"**Query**: {session['analysis_query'][:50]}...")
                    st.write(f"**Date**: {session['created_at'][:16]}")
                    st.caption(f"📁 {session['session_path']}")
            
            if len(previous_sessions) > 5:
                st.caption(f"... and {len(previous_sessions) - 5} more sessions")
        else:
            st.info("No previous sessions found")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Dataset Selection")
        
        # Get available datasets
        datasets = get_available_datasets()
        
        if not datasets:
            st.warning("No datasets found in the /data folder")
            st.info("Available file types: .csv, .json, .tdsx")
            return
        
        # Dataset selection
        selected_dataset = st.selectbox(
            "Choose a dataset:",
            datasets,
            help="Select from available datasets in the /data folder"
        )
        
        # Show file information
        if selected_dataset:
            file_info = get_file_info(selected_dataset)
            if file_info:
                st.subheader("📋 File Information")
                st.write(f"**Size:** {file_info['size_mb']} MB")
                st.write(f"**Modified:** {file_info['modified']}")
                st.write(f"**Type:** {file_info['extension']}")
    
    with col2:
        st.header("💬 Analysis Request")
        
        # Analysis request input
        analysis_query = st.text_area(
            "What would you like to analyze?",
            placeholder="Examples:\n• Show me the trends for last week\n• Find anomalies in the sales data\n• What are the top contributing factors to revenue changes?\n• Analyze customer behavior patterns\n• Identify outliers and their root causes",
            height=150,
            help="Describe what analysis you want to perform. The orchestrator will coordinate the appropriate agents."
        )
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Show me the trends for the last month",
            "Find anomalies and their root causes",
            "Analyze sales performance by region",
            "What factors drive revenue changes?",
            "Identify top and bottom contributors",
            "Show quarterly trends and seasonality"
        ]
        
        selected_example = st.selectbox("Or select an example:", [""] + example_queries)
        if selected_example and st.button("Use Example"):
            analysis_query = selected_example
            st.rerun()
    
    # Action buttons
    st.header("🚀 Execute Analysis")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button(
            "🔍 Start Analysis",
            disabled=not (selected_dataset and analysis_query),
            help="Start the multi-agent analysis pipeline"
        )
    
    with col2:
        if st.button("🔄 Clear Results"):
            if "analysis_results" in st.session_state:
                del st.session_state["analysis_results"]
            st.rerun()
    
    # Execute analysis
    if analyze_button and selected_dataset and analysis_query:
        
        # Create session folder for this analysis
        session_folder = output_manager.create_session_folder(selected_dataset, analysis_query)
        
        # Show session information
        st.info(f"📁 Created analysis session: `{session_folder}`")
        
        # Log the start of analysis
        output_manager.save_log(f"Starting analysis for dataset: {selected_dataset}")
        output_manager.save_log(f"Analysis query: {analysis_query}")
        output_manager.save_log(f"Configuration: depth={analysis_depth}, root_cause={include_root_cause}, format={output_format}")
        
        # Prepare analysis configuration
        analysis_config = {
            "analysis_config": {
                "query": analysis_query,
                "include_root_cause": include_root_cause,
                "depth": analysis_depth,
                "output_format": output_format
            }
        }
        
        # Show progress
        st.header("⏳ Analysis in Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing pipeline...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        status_text.text("📁 Starting data loading...")
        progress_bar.progress(20)
        output_manager.save_log("Starting data loading stage")
        
        # Execute pipeline
        with st.spinner("Running multi-agent analysis pipeline..."):
            try:
                file_path = f"data/{selected_dataset}"
                
                status_text.text("🔄 Executing multi-agent pipeline...")
                progress_bar.progress(50)
                output_manager.save_log("Executing orchestrator pipeline")
                
                results = run_async(client.orchestrate_pipeline(file_path, analysis_config))
                
                progress_bar.progress(80)
                status_text.text("💾 Saving results...")
                
                # Save analysis results to session folder
                output_manager.save_analysis_results(results)
                output_manager.save_log(f"Analysis completed with status: {results.get('status', 'unknown')}")
                
                # Extract insights and recommendations for executive summary
                insights = []
                recommendations = []
                
                # Try to extract insights from results
                if "summary" in results and isinstance(results["summary"], dict):
                    for key, value in results["summary"].items():
                        if "insight" in key.lower() or "finding" in key.lower():
                            insights.append(f"{key}: {value}")
                
                # Create some basic recommendations based on the analysis type
                if "trend" in analysis_query.lower():
                    recommendations.append("Monitor trending patterns for early intervention opportunities")
                if "anomal" in analysis_query.lower():
                    recommendations.append("Investigate root causes of detected anomalies")
                if "performance" in analysis_query.lower():
                    recommendations.append("Focus on improving underperforming metrics")
                
                # Default recommendations if none found
                if not recommendations:
                    recommendations = [
                        "Review analysis results for actionable insights",
                        "Consider implementing data monitoring for ongoing analysis",
                        "Share findings with relevant stakeholders"
                    ]
                
                if not insights:
                    insights = [
                        f"Analysis completed successfully on {selected_dataset}",
                        f"Pipeline processed data through {len(results.get('stages', {}))} stages",
                        f"Analysis depth: {analysis_depth}"
                    ]
                
                # Create executive summary
                output_manager.create_executive_summary(results, insights, recommendations)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis completed and saved!")
                
                # Store results in session state along with session info
                st.session_state["analysis_results"] = results
                st.session_state["session_folder"] = str(session_folder)
                st.session_state["session_summary"] = output_manager.get_session_summary()
                
                # Log completion
                output_manager.save_log("Analysis pipeline completed successfully")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                output_manager.save_log(f"Analysis failed with error: {str(e)}")
                progress_bar.progress(0)
                status_text.text("❌ Analysis failed")
    
    # Display results
    if "analysis_results" in st.session_state:
        st.header("📊 Analysis Results")
        
        # Show session information
        if "session_folder" in st.session_state:
            st.subheader("📁 Session Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Session Folder**: `{st.session_state['session_folder']}`")
                
            with col2:
                if "session_summary" in st.session_state:
                    summary = st.session_state["session_summary"]
                    st.metric("Total Files Created", sum(len(files) for files in summary["files_created"].values()))
                    st.metric("Total Size", f"{summary['total_size_mb']} MB")
            
            # Show files created in expandable section
            if "session_summary" in st.session_state:
                with st.expander("📋 Generated Files", expanded=False):
                    summary = st.session_state["session_summary"]
                    
                    for folder_name, files in summary["files_created"].items():
                        if files:
                            st.write(f"**{folder_name.title()}:**")
                            for file_info in files:
                                st.write(f"  • `{file_info['name']}` ({file_info['size_mb']} MB)")
                                st.caption(f"    Path: {file_info['path']}")
        
        # Display pipeline results
        format_pipeline_results(st.session_state["analysis_results"])

if __name__ == "__main__":
    main() 