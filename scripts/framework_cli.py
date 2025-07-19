#!/usr/bin/env python3
"""
Multi-Agent Data Analysis Framework CLI
Complete command-line interface for the enhanced multi-agent framework
Features: Tableau Hyper API, Google Gemini AI, RootCause Analyst (Why-Bot), and Streamlit Dashboard
"""

import asyncio
import httpx
import json
import sys
import argparse
from pathlib import Path

async def trigger_complete_enhanced_pipeline(file_path: str, config: dict = None):
    """
    Trigger the complete enhanced multi-agent pipeline with comprehensive analysis, 
    root cause investigation, and executive reporting.
    """
    print("🚀 Complete Enhanced Multi-Agent Data Analysis Pipeline")
    print("=" * 70)
    print(f"📁 Processing: {file_path}")
    print("🔧 Complete enhanced capabilities:")
    print("   ⚡ Tableau Hyper API for blazing fast TDSX/Hyper loading")
    print("   🔬 Comprehensive business intelligence analysis suite")
    print("   🔍 RootCause Analyst (Why-Bot) with AI hypothesis generation")
    print("   🤖 Google Gemini AI executive insights and reporting")
    print("   ⚠️ Automatic escalation for low-confidence findings")
    print()
    
    orchestrator_url = "http://localhost:10000"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "orchestrate_pipeline",
        "params": {
            "file_path": file_path,
            "pipeline_config": config or {}
        },
        "id": 1
    }
    
    try:
        print("🔄 Initiating complete enhanced pipeline...")
        timeout = httpx.Timeout(2400.0)  # 40 minutes for complex root cause analysis
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(orchestrator_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                print(f"❌ Pipeline error: {result['error']}")
                return False
            
            pipeline_result = result.get("result", {})
            
            if pipeline_result.get("status") == "completed":
                print("\n🎉 Complete Enhanced Pipeline Finished Successfully!")
                print("=" * 70)
                
                # Display enhanced pipeline summary
                summary = pipeline_result.get("pipeline_summary", {})
                stages = pipeline_result.get("stages", {})
                
                print(f"📊 Pipeline Summary:")
                print(f"   • Total Stages: {summary.get('total_stages', 'N/A')} (Including Root Cause Analysis)")
                print(f"   • Successful Stages: {summary.get('successful_stages', 'N/A')}")
                print(f"   • AI Enhanced: {'✅' if summary.get('ai_enhanced') else '❌'}")
                print(f"   • Hyper API Used: {'⚡' if summary.get('hyper_api_used') else '📁'}")
                print(f"   • Gemini AI Used: {'🤖' if summary.get('gemini_ai_used') else '📊'}")
                print(f"   • Root Cause Completed: {'✅' if summary.get('rootcause_analysis_completed') else '❌'}")
                print(f"   • Requires Escalation: {'⚠️ YES' if summary.get('requires_escalation') else '✅ NO'}")
                print()
                
                # Display detailed stage information
                print("🔧 Stage Details:")
                stage_order = [
                    ("data_loading", "Data Loading"),
                    ("data_cleaning", "Data Cleaning"), 
                    ("data_enrichment", "Data Enrichment"),
                    ("comprehensive_analysis", "Comprehensive Analysis"),
                    ("root_cause_analysis", "Root Cause Analysis (Why-Bot)"),
                    ("executive_reporting", "Executive Reporting")
                ]
                
                for stage_key, stage_name in stage_order:
                    if stage_key in stages:
                        stage_info = stages[stage_key]
                        status_icon = "✅" if stage_info.get("status") == "completed" else "⚠️"
                        print(f"   {status_icon} {stage_name}")
                        
                        # Stage-specific details
                        if stage_key == "data_loading" and stage_info.get("fast_loading_used"):
                            print("      ⚡ Tableau Hyper API fast loading enabled")
                        elif stage_key == "comprehensive_analysis":
                            modules = stage_info.get("analysis_modules_used", 0)
                            print(f"      🔬 {modules} analysis modules executed")
                        elif stage_key == "root_cause_analysis":
                            if stage_info.get("status") == "completed":
                                ai_enhanced = stage_info.get("ai_enhanced", False)
                                print(f"      {'🤖' if ai_enhanced else '📊'} {'AI-powered' if ai_enhanced else 'Statistical'} hypothesis generation")
                                hypotheses = stage_info.get("summary", {}).get("hypotheses_tested", 0)
                                significant = stage_info.get("summary", {}).get("significant_findings", 0)
                                print(f"      🧪 {hypotheses} hypotheses tested, {significant} significant findings")
                                if stage_info.get("requires_escalation"):
                                    print("      ⚠️ Low confidence - escalation recommended")
                                else:
                                    print("      ✅ High confidence analysis")
                            else:
                                print("      ❌ Root cause analysis failed")
                        elif stage_key == "executive_reporting":
                            if stage_info.get("ai_powered"):
                                print("      🤖 Google Gemini AI insights generated")
                            insights = stage_info.get("summary", {}).get("insights_count", "N/A")
                            recommendations = stage_info.get("summary", {}).get("recommendations_count", "N/A")
                            print(f"      📈 Insights: {insights}, Recommendations: {recommendations}")
                            if stage_info.get("includes_rootcause"):
                                print("      🔍 Root cause findings integrated into report")
                print()
                
                # Display results access information
                final_report = pipeline_result.get("final_report_handle_id")
                analysis_handle = stages.get("comprehensive_analysis", {}).get("analysis_handle_id")
                investigation_handle = pipeline_result.get("investigation_handle_id")
                
                print("📄 Access Your Complete Results:")
                print(f"   🎯 Executive Report (Complete): {final_report}")
                print(f"   🔬 Comprehensive Analysis: {analysis_handle}")
                if investigation_handle:
                    print(f"   🔍 Root Cause Investigation: {investigation_handle}")
                else:
                    print("   🔍 Root Cause Investigation: Not completed")
                print()
                
                print("🔧 Extract Reports:")
                print(f"   python extract_report.py {final_report}")
                print(f"   python extract_report.py {analysis_handle}")
                if investigation_handle:
                    print(f"   python extract_report.py {investigation_handle}")
                print()
                
                print("📊 Complete Analysis Features:")
                if summary.get('gemini_ai_used'):
                    print("   🤖 AI-generated executive summary")
                    print("   📈 Strategic business insights")
                    print("   🎯 Data-driven recommendations")
                    print("   📋 Priority action items")
                else:
                    print("   📊 Comprehensive data analysis")
                    print("   📈 Statistical insights")
                    print("   🔍 Pattern detection")
                
                if summary.get('rootcause_analysis_completed'):
                    print("   🔍 Root cause investigation with AI hypotheses")
                    print("   🧪 Statistical hypothesis testing")
                    print("   📊 Variance decomposition analysis")
                    print("   🔗 Causal inference and relationship mapping")
                    print("   💡 Why-Bot automated root cause discovery")
                
                # Escalation warning if needed
                if summary.get('requires_escalation'):
                    print()
                    print("⚠️ ESCALATION REQUIRED:")
                    print("   📞 Manual expert review recommended")
                    print("   🔍 Low confidence in automated root cause analysis")
                    print("   💡 Consider additional data collection or domain expertise")
                else:
                    print()
                    print("✅ ANALYSIS COMPLETE:")
                    print("   🎯 High confidence automated analysis")
                    print("   📊 Ready for business decision making")
                    print("   🚀 Comprehensive insights and recommendations available")
                
                return True
            else:
                print(f"❌ Pipeline failed: {pipeline_result}")
                return False
                
    except httpx.TimeoutException:
        print("⏰ Pipeline timed out - complex root cause analysis may take time")
        print("💡 Check agent logs for progress or consider breaking down the analysis")
        return False
    except Exception as e:
        print(f"❌ Error running complete enhanced pipeline: {e}")
        return False

async def check_complete_enhanced_capabilities():
    """Check if all enhanced capabilities including RootCause Analyst are available."""
    print("🔍 Checking Complete Enhanced Capabilities...")
    print("=" * 60)
    
    agents = [
        ("Data Loader", "http://localhost:10006", "hyper_api_support"),
        ("Data Analyst", "http://localhost:10007", "version"),
        ("RootCause Analyst", "http://localhost:10011", "ai_capabilities"),
        ("Presentation", "http://localhost:10010", "gemini_ai_support"),
        ("Orchestrator", "http://localhost:10000", None)
    ]
    
    all_healthy = True
    
    for agent_name, url, capability_key in agents:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    status = "✅ Healthy"
                    
                    # Check for enhanced capabilities
                    if capability_key and health_data.get(capability_key):
                        if capability_key == "hyper_api_support":
                            status += " ⚡ Hyper API"
                        elif capability_key == "gemini_ai_support":
                            status += " 🤖 Gemini AI"
                        elif capability_key == "ai_capabilities":
                            ai_cap = health_data.get("ai_capabilities")
                            status += f" 🤖 {ai_cap}" if ai_cap else " 🔍 Why-Bot"
                        elif capability_key == "version" and health_data.get("version") == "enhanced":
                            status += " 🔬 Enhanced"
                    
                    # Special handling for RootCause Analyst features
                    if agent_name == "RootCause Analyst":
                        features = health_data.get("features", [])
                        if features:
                            feature_icons = {
                                "ai_hypothesis_generation": "🧠",
                                "statistical_testing": "🧪", 
                                "variance_decomposition": "📊",
                                "causal_inference": "🔗",
                                "confidence_scoring": "💯",
                                "escalation_logic": "⚠️"
                            }
                            feature_summary = " ".join([feature_icons.get(f, "✓") for f in features[:3]])
                            status += f" {feature_summary}"
                    
                    print(f"   {agent_name} Agent: {status}")
                else:
                    print(f"   {agent_name} Agent: ❌ Unhealthy")
                    all_healthy = False
        except Exception as e:
            print(f"   {agent_name} Agent: ❌ Unreachable")
            all_healthy = False
    
    print()
    if all_healthy:
        print("🎉 All complete enhanced capabilities are ready!")
        print("🚀 Ready for comprehensive business intelligence analysis!")
    else:
        print("⚠️ Some agents may need to be restarted with enhanced capabilities")
        print("💡 Run: ./start_all_agents.ps1")
        print("💡 Install enhancements: python install_enhancements.py")
    
    return all_healthy

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Complete Enhanced Multi-Agent Data Analysis with Root Cause Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tdsx_cli.py data.csv                    # Complete enhanced pipeline 
  python tdsx_cli.py dataset.tdsx                # Fast TDSX + root cause analysis
  python tdsx_cli.py data.hyper                  # Direct Hyper file + full analysis
  python tdsx_cli.py --check                     # Check all capabilities
  
Complete Enhanced Features:
  ⚡ Tableau Hyper API - Blazing fast TDSX/Hyper file loading
  🔬 Comprehensive Analysis - 10+ business intelligence modules  
  🔍 RootCause Analyst (Why-Bot) - AI-powered hypothesis generation & testing
  🤖 Google Gemini AI - Executive insights and strategic recommendations
  📊 Professional Reports - Executive-ready presentations with root cause findings
  ⚠️ Smart Escalation - Automatic flagging of low-confidence analyses
        """
    )
    
    parser.add_argument(
        "file_path", 
        nargs="?",
        help="Path to data file (CSV, TDSX, Hyper, JSON, Excel)"
    )
    
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check complete enhanced capabilities status"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="JSON configuration for pipeline customization"
    )
    
    args = parser.parse_args()
    
    if args.check:
        asyncio.run(check_complete_enhanced_capabilities())
        return
    
    if not args.file_path:
        parser.print_help()
        print("\n❌ Error: file_path is required")
        sys.exit(1)
    
    # Parse config if provided
    config = {}
    if args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON config: {e}")
            sys.exit(1)
    
    # Check if file exists
    file_path = Path(args.file_path)
    if not file_path.exists() and not file_path.is_absolute():
        # Try common locations
        test_locations = [
            Path("test_data") / file_path.name,
            Path("data-loader-agent/data") / file_path.name
        ]
        
        found = False
        for location in test_locations:
            if location.exists():
                file_path = location
                found = True
                break
        
        if not found:
            print(f"❌ File not found: {args.file_path}")
            print("💡 Searched locations:")
            for location in test_locations:
                print(f"   - {location}")
            sys.exit(1)
    
    # Run complete enhanced pipeline
    try:
        success = asyncio.run(trigger_complete_enhanced_pipeline(str(file_path), config))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 