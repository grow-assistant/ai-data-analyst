#!/usr/bin/env python3
"""
Test script for the enhanced Schema Profiler Agent with AI capabilities
Demonstrates intelligent profiling and configuration caching
"""

import asyncio
import httpx
import pandas as pd
from pathlib import Path
import json
import sys

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager


async def test_schema_profiler_ai():
    """Test the AI-powered schema profiler capabilities."""
    
    print("🤖 Testing AI-Powered Schema Profiler Agent")
    print("=" * 50)
    
    # Schema profiler agent endpoint
    schema_profiler_url = "http://localhost:10012"
    
    # Test health check
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{schema_profiler_url}/health")
            if response.status_code == 200:
                print("✅ Schema Profiler Agent is healthy")
            else:
                print("❌ Schema Profiler Agent health check failed")
                return
    except Exception as e:
        print(f"❌ Cannot connect to Schema Profiler Agent: {e}")
        return
    
    # Get capabilities
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{schema_profiler_url}/capabilities")
            capabilities = response.json()
            print(f"📋 Available skills: {', '.join(capabilities['skills'])}")
    except Exception as e:
        print(f"⚠️ Could not get capabilities: {e}")
    
    # Test with sample dataset
    data_manager = get_data_handle_manager()
    
    # Create sample sales data
    sample_data = pd.DataFrame({
        'order_id': range(1, 1001),
        'order_date': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'customer_id': [f"CUST_{i%100:03d}" for i in range(1000)],
        'product_category': ['Electronics', 'Clothing', 'Books', 'Home'] * 250,
        'product_name': [f"Product_{i%50}" for i in range(1000)],
        'revenue': [round(50 + (i % 200) * 2.5, 2) for i in range(1000)],
        'quantity': [1 + (i % 5) for i in range(1000)],
        'country': ['USA', 'Canada', 'UK', 'Germany'] * 250,
        'sales_rep': [f"Rep_{i%20}" for i in range(1000)],
        'is_premium': [i % 3 == 0 for i in range(1000)]
    })
    
    print(f"\n📊 Created sample dataset: {len(sample_data)} rows, {len(sample_data.columns)} columns")
    
    # Create data handle
    handle = data_manager.create_handle(
        data=sample_data,
        data_type="dataframe",
        metadata={
            "original_filename": "sample_sales_data.csv",
            "source": "test_script",
            "description": "Sample sales data for testing AI profiler"
        }
    )
    
    print(f"📁 Created data handle: {handle.handle_id}")
    
    # Test 1: List existing configurations
    print(f"\n🔍 Test 1: Checking existing configurations...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "list_all_configs",
            "params": {},
            "id": 1
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(schema_profiler_url, json=payload)
            result = response.json()
            
            if "result" in result:
                configs = result["result"]["configs"]
                print(f"   Found {len(configs)} existing configurations:")
                for config in configs[:3]:  # Show first 3
                    print(f"   • {config['dataset_name']} ({config['dataset_type']}) - {config['configuration_date'][:10]}")
                if len(configs) > 3:
                    print(f"   ... and {len(configs) - 3} more")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"   ❌ Error listing configurations: {e}")
    
    # Test 2: Check for cached configuration of our dataset
    print(f"\n🔍 Test 2: Checking for cached configuration...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "get_dataset_config",
            "params": {
                "data_handle_id": handle.handle_id,
                "dataset_name": "sample_sales_data"
            },
            "id": 2
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(schema_profiler_url, json=payload)
            result = response.json()
            
            if "result" in result:
                cached_config = result["result"].get("cached_config")
                if cached_config:
                    summary = result["result"]["summary"]
                    print(f"   ✅ Found cached configuration!")
                    print(f"      Dataset Type: {summary['dataset_type']}")
                    print(f"      Domain: {summary['domain']}")
                    print(f"      Metrics: {summary['total_metrics']}")
                    print(f"      Dimensions: {summary['total_dimensions']}")
                else:
                    print(f"   📝 No cached configuration found - will generate new one")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"   ❌ Error checking cached config: {e}")
    
    # Test 3: AI-powered profiling
    print(f"\n🤖 Test 3: Running AI-powered profiling...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "ai_profile_dataset",
            "params": {
                "data_handle_id": handle.handle_id,
                "use_cache": True,
                "force_ai": False
            },
            "id": 3
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for AI processing
            response = await client.post(schema_profiler_url, json=payload)
            result = response.json()
            
            if "result" in result:
                profile_result = result["result"]
                summary = profile_result["summary"]
                
                print(f"   ✅ AI Profiling completed!")
                print(f"      Status: {profile_result['status']}")
                print(f"      Dataset Type: {summary['dataset_type']}")
                print(f"      Domain: {summary['domain']}")
                print(f"      Total Rows: {summary['total_rows']:,}")
                print(f"      Total Columns: {summary['total_columns']}")
                print(f"      Dimensions Found: {summary['dimensions_found']}")
                print(f"      Measures Found: {summary['measures_found']}")
                print(f"      Recommendations: {summary['recommendations']}")
                print(f"      Used Cache: {summary.get('cached', False)}")
                
                # Show some specific insights if available
                profile_data = profile_result.get("profile_data", {})
                dimensions = profile_data.get("dimensions", [])
                measures = profile_data.get("measures", [])
                
                if dimensions:
                    print(f"\n   📊 Key Dimensions:")
                    for dim in dimensions[:3]:
                        print(f"      • {dim['column']} ({dim['type']}) - {dim.get('importance', 'unknown')} importance")
                
                if measures:
                    print(f"\n   📈 Key Measures:")
                    for measure in measures[:3]:
                        print(f"      • {measure['column']} ({measure.get('type', 'unknown')}) - {measure.get('business_meaning', 'No description')}")
                
                # Show recommendations
                recommendations = profile_data.get("analysis_recommendations", [])
                if recommendations:
                    print(f"\n   💡 Analysis Recommendations:")
                    for rec in recommendations[:2]:
                        print(f"      • {rec.get('type', 'Analysis')}: {rec.get('description', 'No description')[:80]}...")
                
            else:
                error = result.get('error', {})
                print(f"   ❌ AI Profiling failed: {error.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"   ❌ Error in AI profiling: {e}")
    
    # Test 4: Basic profiling for comparison
    print(f"\n📊 Test 4: Running basic profiling for comparison...")
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "profile_dataset",
            "params": {
                "data_handle_id": handle.handle_id,
                "profile_type": "comprehensive"
            },
            "id": 4
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(schema_profiler_url, json=payload)
            result = response.json()
            
            if "result" in result:
                profile_result = result["result"]
                summary = profile_result["summary"]
                
                print(f"   ✅ Basic Profiling completed!")
                print(f"      Data Quality Score: {summary['data_quality_score']}")
                print(f"      Total Columns: {summary['total_columns']}")
                print(f"      Total Rows: {summary['total_rows']:,}")
                print(f"      Data Types: {summary['data_types']}")
                
            else:
                error = result.get('error', {})
                print(f"   ❌ Basic Profiling failed: {error.get('message', 'Unknown error')}")
                
    except Exception as e:
        print(f"   ❌ Error in basic profiling: {e}")
    
    print(f"\n🎉 Schema Profiler AI Test Complete!")
    print(f"💡 The AI profiler can now intelligently analyze datasets and cache configurations for reuse!")


if __name__ == "__main__":
    print("🚀 Starting Schema Profiler AI Test...")
    print("📝 Make sure the schema-profiler-agent is running on port 10012")
    print("🔑 Ensure GOOGLE_API_KEY is set for AI features\n")
    
    asyncio.run(test_schema_profiler_ai()) 