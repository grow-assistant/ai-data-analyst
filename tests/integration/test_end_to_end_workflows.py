#!/usr/bin/env python3
"""
Phase 4: End-to-End Workflow Testing
Tests complete workflow scenarios, scheduling, and production-like use cases.
"""

import asyncio
import httpx
import tempfile
import pandas as pd
import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndToEndWorkflowTester:
    """End-to-end workflow testing suite."""
    
    def __init__(self):
        self.base_urls = {
            'orchestrator': 'http://localhost:8000',
            'data_loader': 'http://localhost:10006',
            'data_cleaning': 'http://localhost:10008',
            'data_enrichment': 'http://localhost:10009',
            'data_analyst': 'http://localhost:10007',
            'presentation': 'http://localhost:10010'
        }
        self.test_results = {}
        self.workflow_results = []
        
    def create_daily_sales_report_scenario(self) -> pd.DataFrame:
        """Create realistic daily sales data for workflow testing."""
        # Simulate daily sales data with realistic patterns
        base_date = datetime.now() - timedelta(days=30)
        data = []
        
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            # Simulate varying daily sales volumes
            daily_transactions = 10 + (day % 7) * 5  # Weekly pattern
            
            for transaction in range(daily_transactions):
                # Realistic sales data with some quality issues
                sales_amount = 50 + (transaction * 15) + (day * 5)
                if day % 10 == 0 and transaction % 5 == 0:  # Some missing values
                    sales_amount = None
                    
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'time': f"{9 + (transaction % 8)}:00:00",
                    'transaction_id': f"TXN_{day:03d}_{transaction:03d}",
                    'customer_id': f"CUST_{(transaction * 7) % 100:03d}",
                    'product_category': ['Electronics', 'Clothing', 'Books', 'Home'][transaction % 4],
                    'product_name': f"Product_{(transaction * 3) % 50:02d}",
                    'sales_amount': sales_amount,
                    'quantity': 1 + (transaction % 3),
                    'region': ['North', 'South', 'East', 'West'][day % 4],
                    'salesperson': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'][transaction % 5]
                })
        
        return pd.DataFrame(data)
    
    def create_financial_analysis_scenario(self) -> pd.DataFrame:
        """Create financial data for comprehensive analysis workflow."""
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        years = [2022, 2023, 2024]
        departments = ['Sales', 'Marketing', 'R&D', 'Operations', 'HR']
        
        data = []
        for year in years:
            for quarter in quarters:
                for dept in departments:
                    # Simulate quarterly financial data
                    base_budget = 100000 + (hash(dept) % 50000)
                    seasonal_factor = 1.2 if quarter in ['Q4', 'Q1'] else 1.0
                    
                    revenue = base_budget * seasonal_factor * (1 + (year - 2022) * 0.1)
                    expenses = revenue * (0.7 + (hash(dept + quarter) % 20) / 100)
                    
                    data.append({
                        'year': year,
                        'quarter': quarter,
                        'department': dept,
                        'revenue': revenue,
                        'expenses': expenses,
                        'profit': revenue - expenses,
                        'budget_variance': (revenue - base_budget) / base_budget * 100,
                        'employee_count': 10 + (hash(dept) % 20),
                        'efficiency_score': 75 + (hash(dept + str(year)) % 25)
                    })
        
        return pd.DataFrame(data)
    
    async def execute_workflow(self, data: pd.DataFrame, workflow_name: str, 
                             workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete workflow with specified configuration."""
        logger.info(f"🚀 Executing workflow: {workflow_name}")
        
        workflow_start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            test_file = f.name
        
        workflow_result = {
            'workflow_name': workflow_name,
            'start_time': datetime.now().isoformat(),
            'input_records': len(data),
            'stages': {},
            'success': False,
            'execution_time': 0,
            'data_handles': [],
            'configuration': workflow_config
        }
        
        try:
            # Stage 1: Data Loading
            logger.info(f"📂 Stage 1: Loading {workflow_name} data...")
            load_start = time.time()
            
            load_payload = {
                "jsonrpc": "2.0",
                "method": "load_dataset",
                "params": {"file_path": test_file},
                "id": f"workflow_{workflow_name}_load"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_loader'], json=load_payload, timeout=30.0)
                load_result = response.json()
            
            if "result" not in load_result:
                workflow_result['stages']['loading'] = {
                    'status': 'FAILED', 
                    'error': load_result.get('error'),
                    'duration': time.time() - load_start
                }
                return workflow_result
            
            data_handle_id = load_result["result"]["data_handle_id"]
            workflow_result['stages']['loading'] = {
                'status': 'SUCCESS',
                'data_handle_id': data_handle_id,
                'duration': time.time() - load_start,
                'records_loaded': len(data)
            }
            workflow_result['data_handles'].append(data_handle_id)
            
            # Stage 2: Data Cleaning (if enabled)
            if workflow_config.get('enable_cleaning', True):
                logger.info(f"🧹 Stage 2: Cleaning {workflow_name} data...")
                clean_start = time.time()
                
                clean_payload = {
                    "jsonrpc": "2.0",
                    "method": "clean_dataset",
                    "params": {
                        "data_handle_id": data_handle_id,
                        "operations": workflow_config.get('cleaning_operations', 
                                                        ["remove_duplicates", "handle_missing_values"])
                    },
                    "id": f"workflow_{workflow_name}_clean"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.base_urls['data_cleaning'], json=clean_payload, timeout=30.0)
                    clean_result = response.json()
                
                if "result" not in clean_result:
                    workflow_result['stages']['cleaning'] = {
                        'status': 'FAILED', 
                        'error': clean_result.get('error'),
                        'duration': time.time() - clean_start
                    }
                    return workflow_result
                
                cleaned_handle_id = clean_result["result"]["cleaned_data_handle_id"]
                workflow_result['stages']['cleaning'] = {
                    'status': 'SUCCESS',
                    'cleaned_data_handle_id': cleaned_handle_id,
                    'duration': time.time() - clean_start,
                    'operations_performed': clean_result["result"].get('operations', [])
                }
                workflow_result['data_handles'].append(cleaned_handle_id)
                current_handle = cleaned_handle_id
            else:
                current_handle = data_handle_id
                workflow_result['stages']['cleaning'] = {'status': 'SKIPPED'}
            
            # Stage 3: Data Enrichment (if enabled)
            if workflow_config.get('enable_enrichment', True):
                logger.info(f"🔄 Stage 3: Enriching {workflow_name} data...")
                enrich_start = time.time()
                
                enrich_payload = {
                    "jsonrpc": "2.0",
                    "method": "enrich_dataset",
                    "params": {"data_handle_id": current_handle},
                    "id": f"workflow_{workflow_name}_enrich"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(self.base_urls['data_enrichment'], json=enrich_payload, timeout=30.0)
                    enrich_result = response.json()
                
                if "result" not in enrich_result:
                    workflow_result['stages']['enrichment'] = {
                        'status': 'FAILED', 
                        'error': enrich_result.get('error'),
                        'duration': time.time() - enrich_start
                    }
                    return workflow_result
                
                enriched_handle_id = enrich_result["result"]["enriched_data_handle_id"]
                workflow_result['stages']['enrichment'] = {
                    'status': 'SUCCESS',
                    'enriched_data_handle_id': enriched_handle_id,
                    'duration': time.time() - enrich_start
                }
                workflow_result['data_handles'].append(enriched_handle_id)
                current_handle = enriched_handle_id
            else:
                workflow_result['stages']['enrichment'] = {'status': 'SKIPPED'}
            
            # Stage 4: Data Analysis
            logger.info(f"📊 Stage 4: Analyzing {workflow_name} data...")
            analyze_start = time.time()
            
            analyze_payload = {
                "jsonrpc": "2.0",
                "method": "analyze_dataset",
                "params": {"data_handle_id": current_handle},
                "id": f"workflow_{workflow_name}_analyze"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['data_analyst'], json=analyze_payload, timeout=30.0)
                analyze_result = response.json()
            
            if "result" not in analyze_result:
                workflow_result['stages']['analysis'] = {
                    'status': 'FAILED', 
                    'error': analyze_result.get('error'),
                    'duration': time.time() - analyze_start
                }
                return workflow_result
            
            workflow_result['stages']['analysis'] = {
                'status': 'SUCCESS',
                'duration': time.time() - analyze_start,
                'analysis_results': analyze_result["result"]
            }
            
            # Stage 5: Report Generation
            logger.info(f"📄 Stage 5: Creating {workflow_name} report...")
            report_start = time.time()
            
            report_payload = {
                "jsonrpc": "2.0",
                "method": "create_report",
                "params": {"data_handle_id": current_handle},
                "id": f"workflow_{workflow_name}_report"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_urls['presentation'], json=report_payload, timeout=30.0)
                report_result = response.json()
            
            if "result" not in report_result:
                workflow_result['stages']['presentation'] = {
                    'status': 'FAILED', 
                    'error': report_result.get('error'),
                    'duration': time.time() - report_start
                }
                return workflow_result
            
            workflow_result['stages']['presentation'] = {
                'status': 'SUCCESS',
                'duration': time.time() - report_start,
                'report_details': report_result["result"]
            }
            
            # Mark overall success
            workflow_result['success'] = True
            workflow_result['execution_time'] = time.time() - workflow_start_time
            workflow_result['end_time'] = datetime.now().isoformat()
            
            logger.info(f"🎉 Workflow {workflow_name} completed successfully in {workflow_result['execution_time']:.2f}s")
            
            return workflow_result
            
        except Exception as e:
            workflow_result['error'] = str(e)
            workflow_result['execution_time'] = time.time() - workflow_start_time
            logger.error(f"❌ Workflow {workflow_name} failed: {e}")
            return workflow_result
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    async def test_daily_sales_workflow(self):
        """Test automated daily sales reporting workflow."""
        logger.info("🧪 Testing Daily Sales Reporting Workflow")
        
        sales_data = self.create_daily_sales_report_scenario()
        
        workflow_config = {
            'enable_cleaning': True,
            'enable_enrichment': True,
            'cleaning_operations': ['remove_duplicates', 'handle_missing_values'],
            'analysis_type': 'sales_summary',
            'report_format': 'daily_summary'
        }
        
        result = await self.execute_workflow(sales_data, 'daily_sales_report', workflow_config)
        self.test_results['daily_sales_workflow'] = result
        self.workflow_results.append(result)
        
        return result
    
    async def test_financial_analysis_workflow(self):
        """Test comprehensive financial analysis workflow."""
        logger.info("🧪 Testing Financial Analysis Workflow")
        
        financial_data = self.create_financial_analysis_scenario()
        
        workflow_config = {
            'enable_cleaning': True,
            'enable_enrichment': True,
            'cleaning_operations': ['remove_duplicates', 'handle_missing_values'],
            'analysis_type': 'financial_analysis',
            'report_format': 'executive_summary'
        }
        
        result = await self.execute_workflow(financial_data, 'financial_analysis', workflow_config)
        self.test_results['financial_analysis_workflow'] = result
        self.workflow_results.append(result)
        
        return result
    
    async def test_custom_workflow_scenarios(self):
        """Test various custom workflow configurations."""
        logger.info("🧪 Testing Custom Workflow Scenarios")
        
        # Test 1: Minimal workflow (cleaning disabled)
        minimal_data = self.create_daily_sales_report_scenario().head(20)
        minimal_config = {
            'enable_cleaning': False,
            'enable_enrichment': True,
            'analysis_type': 'quick_summary',
            'report_format': 'brief'
        }
        
        minimal_result = await self.execute_workflow(minimal_data, 'minimal_workflow', minimal_config)
        self.test_results['minimal_workflow'] = minimal_result
        self.workflow_results.append(minimal_result)
        
        # Test 2: Analysis-only workflow (no enrichment)
        analysis_data = self.create_financial_analysis_scenario().head(30)
        analysis_config = {
            'enable_cleaning': True,
            'enable_enrichment': False,
            'analysis_type': 'statistical_only',
            'report_format': 'data_focused'
        }
        
        analysis_result = await self.execute_workflow(analysis_data, 'analysis_only_workflow', analysis_config)
        self.test_results['analysis_only_workflow'] = analysis_result
        self.workflow_results.append(analysis_result)
        
        return [minimal_result, analysis_result]
    
    async def test_large_dataset_workflow(self):
        """Test workflow performance with larger datasets."""
        logger.info("🧪 Testing Large Dataset Workflow Performance")
        
        # Create larger dataset (300+ records)
        large_sales_data = self.create_daily_sales_report_scenario()
        # Duplicate data to make it larger
        large_data = pd.concat([large_sales_data] * 3, ignore_index=True)
        
        large_config = {
            'enable_cleaning': True,
            'enable_enrichment': True,
            'cleaning_operations': ['remove_duplicates', 'handle_missing_values'],
            'analysis_type': 'comprehensive',
            'report_format': 'detailed'
        }
        
        result = await self.execute_workflow(large_data, 'large_dataset_workflow', large_config)
        self.test_results['large_dataset_workflow'] = result
        self.workflow_results.append(result)
        
        return result
    
    async def run_all_workflow_tests(self):
        """Run comprehensive end-to-end workflow tests."""
        logger.info("🚀 Starting Phase 4: End-to-End Workflow Testing")
        logger.info("=" * 70)
        
        # Test 1: Daily Sales Workflow
        await self.test_daily_sales_workflow()
        
        # Test 2: Financial Analysis Workflow
        await self.test_financial_analysis_workflow()
        
        # Test 3: Custom Workflow Scenarios
        await self.test_custom_workflow_scenarios()
        
        # Test 4: Large Dataset Performance
        await self.test_large_dataset_workflow()
        
        # Generate comprehensive report
        self.generate_workflow_report()
    
    def generate_workflow_report(self):
        """Generate comprehensive workflow test report."""
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 4: END-TO-END WORKFLOW TEST RESULTS")
        logger.info("=" * 70)
        
        total_workflows = len(self.test_results)
        successful_workflows = sum(1 for result in self.test_results.values() 
                                 if isinstance(result, dict) and result.get('success', False))
        
        logger.info(f"Total Workflows Tested: {total_workflows}")
        logger.info(f"Successful: {successful_workflows}")
        logger.info(f"Failed: {total_workflows - successful_workflows}")
        logger.info(f"Success Rate: {(successful_workflows/total_workflows*100):.1f}%" if total_workflows > 0 else "No workflows run")
        
        # Calculate performance metrics
        total_records_processed = sum(result.get('input_records', 0) for result in self.workflow_results)
        total_execution_time = sum(result.get('execution_time', 0) for result in self.workflow_results)
        total_data_handles = sum(len(result.get('data_handles', [])) for result in self.workflow_results)
        
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"   Total Records Processed: {total_records_processed}")
        logger.info(f"   Total Execution Time: {total_execution_time:.2f} seconds")
        logger.info(f"   Average Processing Rate: {total_records_processed/total_execution_time:.1f} records/second")
        logger.info(f"   Total Data Handles Created: {total_data_handles}")
        
        logger.info("\nDetailed Workflow Results:")
        for workflow_name, result in self.test_results.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                status_emoji = "✅" if success else "❌"
                execution_time = result.get('execution_time', 0)
                records = result.get('input_records', 0)
                
                logger.info(f"   {status_emoji} {workflow_name}: {'SUCCESS' if success else 'FAILED'}")
                logger.info(f"      Records: {records}, Time: {execution_time:.2f}s, Rate: {records/execution_time:.1f}/s")
                
                if success and 'stages' in result:
                    stages_completed = sum(1 for stage in result['stages'].values() 
                                         if isinstance(stage, dict) and stage.get('status') == 'SUCCESS')
                    total_stages = len([s for s in result['stages'].values() if isinstance(s, dict) and s.get('status') != 'SKIPPED'])
                    logger.info(f"      Stages: {stages_completed}/{total_stages} completed")
        
        # Save detailed results
        results_file = f"test_results_workflows_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"\n📊 Detailed results saved to: {results_file}")
        
        if successful_workflows == total_workflows and total_workflows > 0:
            logger.info("\n🎉 ALL END-TO-END WORKFLOW TESTS PASSED!")
            logger.info("✅ Framework ready for production deployment!")
            logger.info("✅ Ready for Phase 5: Security and Observability Testing")
        else:
            logger.info(f"\n⚠️ {total_workflows - successful_workflows} workflows need attention")

async def main():
    """Main workflow test function."""
    tester = EndToEndWorkflowTester()
    await tester.run_all_workflow_tests()

if __name__ == "__main__":
    asyncio.run(main()) 