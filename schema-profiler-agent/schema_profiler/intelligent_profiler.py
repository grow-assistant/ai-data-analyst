"""Intelligent dataset profiler using Google Gemini for AI-powered data analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os
import hashlib
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

# Try to load from .env file in various locations
env_paths = [
    Path.cwd() / '.env',  # Current directory
    Path.cwd().parent / '.env',  # Parent directory
    Path.cwd().parent.parent / '.env',  # Grandparent directory
    Path(__file__).parent.parent.parent / '.env',  # Project root
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI not installed. Install with: pip install google-generativeai")


class IntelligentDatasetProfiler:
    """AI-powered dataset profiler using Google Gemini."""
    
    def __init__(self, api_key: str = None, cache_dir: str = "outputs/configs"):
        """Initialize with Gemini API key and cache directory."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package is required. Install with: pip install google-generativeai")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment (which includes .env file)
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                # Provide helpful error message
                project_root = Path(__file__).parent.parent.parent
                env_file_path = project_root / '.env'
                raise ValueError(
                    f"No Google API key found. Please do one of the following:\n"
                    f"1. Create a .env file at: {env_file_path}\n"
                    f"   with content: GOOGLE_API_KEY=your-api-key-here\n"
                    f"2. Set GOOGLE_API_KEY environment variable\n"
                    f"3. Pass api_key parameter to this function"
                )
        
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate_dataset_hash(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Generate a hash for the dataset based on structure and sample data."""
        # Create a signature of the dataset structure
        structure_info = {
            'columns': list(df.columns),
            'dtypes': [str(dtype) for dtype in df.dtypes],
            'shape': df.shape,
            'dataset_name': dataset_name
        }
        
        # Add sample data hash for consistency
        if len(df) > 0:
            sample_data = df.head(min(100, len(df))).to_string()
            structure_info['sample_hash'] = hashlib.md5(sample_data.encode()).hexdigest()[:8]
        
        # Create hash of the structure
        structure_str = json.dumps(structure_info, sort_keys=True)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    def get_cached_config(self, dataset_name: str, dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Check if a configuration exists for this dataset."""
        config_file = self.cache_dir / f"config_{dataset_name.lower()}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if the hash matches (dataset structure hasn't changed)
                if config.get('dataset_hash') == dataset_hash:
                    print(f"✅ Found cached configuration for {dataset_name}")
                    return config
                else:
                    print(f"⚠️ Dataset structure changed for {dataset_name}, will regenerate config")
                    
            except Exception as e:
                print(f"⚠️ Error reading cached config for {dataset_name}: {e}")
        
        return None
    
    def save_config(self, config: Dict[str, Any], dataset_name: str) -> Path:
        """Save configuration to cache."""
        config_file = self.cache_dir / f"config_{dataset_name.lower()}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"💾 Saved configuration to: {config_file}")
        return config_file
    
    def profile_with_gemini(self, df: pd.DataFrame, dataset_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Profile dataset using Gemini AI to intelligently analyze columns, types, and relationships.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            use_cache: Whether to use cached configurations if available
            
        Returns:
            Complete profiling results with AI-powered insights
        """
        print(f"🤖 Analyzing {dataset_name} with Gemini AI...")
        
        # Check for cached configuration first
        dataset_hash = self.generate_dataset_hash(df, dataset_name)
        
        if use_cache:
            cached_config = self.get_cached_config(dataset_name, dataset_hash)
            if cached_config:
                return self._convert_config_to_profile(cached_config)
        
        # Get basic structure information
        structure = self._get_dataset_structure(df)
        
        # Take a sample of 100 records for Gemini analysis
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Prepare column information
        columns_info = self._prepare_column_info(df)
        
        # Convert sample data to dict for JSON serialization
        sample_records = self._prepare_sample_records(sample_df)
        
        # Create comprehensive prompt for Gemini
        prompt = self._create_comprehensive_prompt(
            dataset_name, 
            structure, 
            columns_info, 
            sample_records
        )
        
        try:
            # Configure generation with JSON schema for consistent output
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more consistent output
                response_mime_type="application/json"
            )
            
            # Get Gemini's analysis with structured output
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse the response
            gemini_analysis = self._parse_gemini_response(response.text)
            
            # Ensure all required fields exist with defaults
            gemini_analysis = self._ensure_complete_analysis(gemini_analysis)
            
            # Create configuration from analysis
            config = self._create_config_from_analysis(df, dataset_name, dataset_hash, gemini_analysis)
            
            # Save configuration for future use
            if use_cache:
                self.save_config(config, dataset_name)
            
            # Combine with basic structure info
            profile_result = {
                'structure': structure,
                'gemini_analysis': gemini_analysis,
                'column_types': gemini_analysis.get('column_types', {}),
                'data_quality': self._assess_data_quality_with_ai(df, gemini_analysis),
                'dimensions': gemini_analysis.get('dimensions', []),
                'measures': gemini_analysis.get('measures', []),
                'relationships': gemini_analysis.get('relationships', []),
                'analysis_recommendations': gemini_analysis.get('analysis_recommendations', []),
                'configuration': config
            }
            
            return profile_result
            
        except Exception as e:
            print(f"❌ Error calling Gemini: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_config_to_profile(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert cached configuration back to profile format."""
        ai_insights = config.get('ai_insights', {})
        column_mappings = config.get('column_mappings', {})
        
        # Reconstruct dimensions and measures from configuration
        dimensions = []
        for dim_col in column_mappings.get('all_dimensions', []):
            dimensions.append({
                'column': dim_col,
                'type': 'categorical',
                'importance': 'high' if dim_col in column_mappings.get('key_dimensions', []) else 'medium'
            })
        
        measures = []
        measure_metadata = config.get('measure_metadata', {})
        for measure_col in column_mappings.get('all_metrics', []):
            measures.append({
                'column': measure_col,
                'type': measure_metadata.get(measure_col, {}).get('type', 'additive'),
                'business_meaning': measure_metadata.get(measure_col, {}).get('business_meaning', ''),
                'importance': 'critical' if measure_col in column_mappings.get('key_metrics', []) else 'medium'
            })
        
        # Create gemini_analysis structure
        gemini_analysis = {
            'dataset_summary': {
                'dataset_type': ai_insights.get('dataset_type', 'Unknown'),
                'domain': ai_insights.get('domain', 'Unknown')
            },
            'column_types': {
                'dates': column_mappings.get('all_dates', []),
                'categories': column_mappings.get('categorical_dimensions', []),
                'measures': column_mappings.get('all_metrics', []),
                'ids': column_mappings.get('id_columns', []),
                'geographical': column_mappings.get('geographical_dimensions', []),
                'hierarchical': column_mappings.get('hierarchical_dimensions', [])
            },
            'dimensions': dimensions,
            'measures': measures,
            'relationships': config.get('relationships', []),
            'analysis_recommendations': ai_insights.get('recommendations', [])
        }
        
        return {
            'structure': {'from_cache': True},
            'gemini_analysis': gemini_analysis,
            'column_types': gemini_analysis['column_types'],
            'dimensions': dimensions,
            'measures': measures,
            'relationships': config.get('relationships', []),
            'analysis_recommendations': ai_insights.get('recommendations', []),
            'configuration': config,
            'cached': True
        }
    
    def _create_config_from_analysis(self, df: pd.DataFrame, dataset_name: str, 
                                   dataset_hash: str, gemini_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration from Gemini analysis."""
        column_types = gemini_analysis.get('column_types', {})
        primary_keys = gemini_analysis.get('primary_keys', {})
        
        # Build configuration based on Gemini's analysis
        config = {
            'dataset_name': dataset_name,
            'dataset_hash': dataset_hash,
            'configuration_date': datetime.now().isoformat(),
            'configuration_source': 'gemini_intelligent_profiler',
            'column_mappings': {},
            'user_preferences': {
                'analysis_depth': 'comprehensive',
                'auto_spawn_agents': True
            },
            'ai_insights': {
                'dataset_type': gemini_analysis.get('dataset_summary', {}).get('dataset_type'),
                'domain': gemini_analysis.get('dataset_summary', {}).get('domain'),
                'recommendations': gemini_analysis.get('analysis_recommendations', [])
            }
        }
        
        # Configure date columns from Gemini analysis
        all_dates = column_types.get('dates', [])
        config['column_mappings']['primary_date'] = primary_keys.get('primary_date')
        config['column_mappings']['other_dates'] = [d for d in all_dates if d != primary_keys.get('primary_date')]
        config['column_mappings']['all_dates'] = all_dates
        
        # Configure metric columns from Gemini analysis
        all_metrics = [m['column'] for m in gemini_analysis.get('measures', [])]
        config['column_mappings']['primary_metric'] = primary_keys.get('primary_metric')
        config['column_mappings']['secondary_metrics'] = [m for m in all_metrics if m != primary_keys.get('primary_metric')]
        config['column_mappings']['all_metrics'] = all_metrics
        
        # Get top metrics by importance
        important_metrics = [
            m['column'] for m in gemini_analysis.get('measures', [])
            if m.get('importance') in ['critical', 'high']
        ]
        config['column_mappings']['key_metrics'] = important_metrics[:10]
        
        # Configure dimension columns from Gemini analysis
        all_dimensions = [d['column'] for d in gemini_analysis.get('dimensions', [])]
        
        # Categorize dimensions based on Gemini's classification
        geo_dimensions = column_types.get('geographical', [])
        hier_dimensions = column_types.get('hierarchical', [])
        cat_dimensions = column_types.get('categories', [])
        
        config['column_mappings']['geographical_dimensions'] = geo_dimensions
        config['column_mappings']['hierarchical_dimensions'] = hier_dimensions
        config['column_mappings']['categorical_dimensions'] = cat_dimensions
        config['column_mappings']['all_dimensions'] = all_dimensions
        config['column_mappings']['primary_dimension'] = primary_keys.get('primary_dimension')
        
        # Key dimensions based on importance
        key_dims = [
            d['column'] for d in gemini_analysis.get('dimensions', [])
            if d.get('importance') in ['high', 'medium']
        ][:10]
        config['column_mappings']['key_dimensions'] = key_dims
        
        # Configure special columns from Gemini analysis
        config['column_mappings']['id_columns'] = column_types.get('ids', [])
        config['column_mappings']['text_columns'] = column_types.get('text', [])
        config['column_mappings']['binary_columns'] = column_types.get('binary', [])
        
        # Set user preferences based on Gemini's insights
        config['user_preferences']['geographical_analysis'] = len(geo_dimensions) > 0
        config['user_preferences']['time_series_analysis'] = len(all_dates) > 0
        config['user_preferences']['hierarchical_analysis'] = len(hier_dimensions) > 0
        
        # Add measure metadata for enhanced analysis
        config['measure_metadata'] = {
            m['column']: {
                'type': m.get('type', 'additive'),
                'business_meaning': m.get('business_meaning', ''),
                'aggregation_method': m.get('aggregation_method', 'sum'),
                'unit': m.get('unit', 'unknown')
            }
            for m in gemini_analysis.get('measures', [])
        }
        
        # Add relationships identified by Gemini
        config['relationships'] = gemini_analysis.get('relationships', [])
        
        return config
    
    def _get_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset structure information."""
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # Find date columns and get date range
        date_columns = []
        date_range = None
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
                if date_range is None:
                    date_range = {
                        "start": df[col].min().strftime("%Y-%m-%d") if pd.notna(df[col].min()) else None,
                        "end": df[col].max().strftime("%Y-%m-%d") if pd.notna(df[col].max()) else None
                    }
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(memory_usage, 2),
            "column_names": list(df.columns),
            "date_columns": date_columns,
            "date_range": date_range
        }
    
    def _prepare_column_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare column information for Gemini analysis."""
        columns_info = []
        
        # For very large datasets, use sampling for statistics
        use_sample = len(df) > 1_000_000
        if use_sample:
            print(f"  📊 Using sample for statistics (dataset has {len(df):,} rows)")
            df_sample = df.sample(min(100_000, len(df) // 10), random_state=42)
        else:
            df_sample = df
        
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': round(df[col].isna().sum() / len(df) * 100, 2),
                'sample_values': []
            }
            
            # Get sample values, handling different types
            try:
                sample_values = df[col].dropna().head(10)
                if pd.api.types.is_bool_dtype(df[col]):
                    col_info['sample_values'] = [bool(v) for v in sample_values]
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_info['sample_values'] = [v.strftime("%Y-%m-%d %H:%M:%S") for v in sample_values]
                else:
                    col_info['sample_values'] = sample_values.tolist()
            except Exception:
                col_info['sample_values'] = []
            
            # Add statistics for numeric columns (excluding boolean)
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                try:
                    # Use sample for stats if dataset is large
                    col_data = df_sample[col] if use_sample else df[col]
                    col_info.update({
                        'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else 0,
                        'std': float(col_data.std()) if not pd.isna(col_data.std()) else 0,
                        'min': float(col_data.min()) if not pd.isna(col_data.min()) else 0,
                        'max': float(col_data.max()) if not pd.isna(col_data.max()) else 0,
                        'q1': float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else 0,
                        'q3': float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else 0
                    })
                except Exception as e:
                    # If quantile calculation fails, skip those stats
                    print(f"  ⚠️ Warning: Could not calculate statistics for column '{col}': {str(e)}")
                    col_info.update({
                        'mean': 0,
                        'std': 0,
                        'min': 0,
                        'max': 0,
                        'q1': 0,
                        'q3': 0
                    })
            elif pd.api.types.is_bool_dtype(df[col]):
                # For boolean columns, provide meaningful stats
                col_data = df_sample[col] if use_sample else df[col]
                true_count = col_data.sum()
                total_count = len(col_data)
                false_count = total_count - true_count
                col_info.update({
                    'true_count': int(true_count),
                    'false_count': int(false_count),
                    'true_percentage': round(true_count / total_count * 100, 2)
                })
            
            columns_info.append(col_info)
        
        return columns_info
    
    def _prepare_sample_records(self, sample_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare sample records for Gemini, handling various data types."""
        sample_records = []
        
        for _, row in sample_df.iterrows():
            record = {}
            for col, value in row.items():
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    record[col] = value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value)
                else:
                    record[col] = str(value)
            sample_records.append(record)
        
        return sample_records
    
    def _create_comprehensive_prompt(self, dataset_name: str, structure: Dict,
                                   columns_info: List[Dict], sample_records: List[Dict]) -> str:
        """Create a comprehensive prompt for Gemini to analyze the dataset."""
        
        prompt = f"""You are an expert data analyst. Analyze this dataset comprehensively and provide detailed insights.

DATASET: {dataset_name}
SHAPE: {structure['total_rows']} rows × {structure['total_columns']} columns

COLUMN INFORMATION:
{json.dumps(columns_info, indent=2)}

SAMPLE RECORDS (up to 100 records):
{json.dumps(sample_records, indent=2, default=str)}

Please provide a comprehensive analysis in the following JSON format:
{{
    "dataset_summary": {{
        "dataset_type": "e.g., retail sales, time series, operational metrics, etc.",
        "domain": "e.g., e-commerce, finance, logistics, healthcare, etc.",
        "temporal_nature": "static snapshot, time series, event log, etc.",
        "granularity": "transaction-level, daily aggregates, etc."
    }},
    
    "column_types": {{
        "dates": ["list of date/time columns"],
        "categories": ["list of categorical columns with <50 unique values"],
        "measures": ["list of numeric columns that are business metrics/values"],
        "ids": ["list of high-cardinality identifier columns"],
        "text": ["list of long text/description columns"],
        "binary": ["list of binary/boolean columns"],
        "hierarchical": ["columns that form hierarchies like category/subcategory"],
        "geographical": ["columns with location data like country/state/city"]
    }},
    
    "dimensions": [
        {{
            "column": "column_name",
            "type": "categorical|binary|temporal|geographical|hierarchical",
            "cardinality": "number of unique values",
            "importance": "high|medium|low",
            "reasoning": "why this is a good dimension for analysis"
        }}
    ],
    
    "measures": [
        {{
            "column": "column_name",
            "type": "additive|semi-additive|non-additive|ratio|percentage",
            "business_meaning": "what this metric represents",
            "aggregation_method": "sum|avg|max|min|count",
            "importance": "critical|high|medium|low",
            "unit": "currency|percentage|count|time|distance|etc"
        }}
    ],
    
    "primary_keys": {{
        "primary_date": "main date column for time-based analysis",
        "primary_metric": "most important business metric",
        "primary_dimension": "main grouping dimension",
        "reasoning": "explanation of why these were chosen"
    }},
    
    "relationships": [
        {{
            "type": "hierarchy|correlation|foreign_key|derived",
            "columns": ["column1", "column2"],
            "description": "nature of the relationship",
            "strength": "strong|moderate|weak"
        }}
    ],
    
    "data_quality_insights": {{
        "completeness": "assessment of missing data",
        "consistency": "assessment of data consistency",
        "outliers": "notable outliers or anomalies observed",
        "data_issues": ["list of potential data quality issues"]
    }},
    
    "analysis_recommendations": [
        {{
            "type": "trend_analysis|comparison|segmentation|correlation|anomaly_detection",
            "description": "specific analysis recommendation",
            "metrics": ["relevant metrics for this analysis"],
            "dimensions": ["relevant dimensions for this analysis"],
            "business_value": "what insights this could provide"
        }}
    ],
    
    "advanced_insights": {{
        "seasonality_potential": "whether time-based patterns might exist",
        "segmentation_opportunities": ["ways to segment the data"],
        "kpi_suggestions": ["suggested KPIs based on available data"],
        "dashboard_recommendations": ["key visualizations to create"]
    }}
}}

Guidelines for analysis:
1. Be thorough and identify ALL columns correctly based on their values, not just names
2. For measures, distinguish between different types (additive like sales vs ratios like percentages)
3. Identify natural hierarchies (e.g., Category->Subcategory->Product)
4. Suggest meaningful relationships between columns
5. Provide actionable analysis recommendations based on the data
6. Consider the business context when determining importance
7. Look for derived metrics (e.g., Profit = Sales - Cost)
8. Identify geographical hierarchies (Country->State->City)
9. Assess data quality based on nulls, outliers, and consistency"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response."""
        try:
            # If response is already valid JSON, parse directly
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            
            # Otherwise try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                print("❌ Could not find valid JSON in Gemini response")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {}
    
    def _ensure_complete_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist in the analysis with sensible defaults."""
        # Default structure for complete analysis
        defaults = {
            "dataset_summary": {
                "dataset_type": "Unknown",
                "domain": "Unknown",
                "temporal_nature": "Unknown",
                "granularity": "Unknown"
            },
            "column_types": {
                "dates": [],
                "categories": [],
                "measures": [],
                "ids": [],
                "text": [],
                "binary": [],
                "hierarchical": [],
                "geographical": []
            },
            "dimensions": [],
            "measures": [],
            "primary_keys": {
                "primary_date": None,
                "primary_metric": None,
                "primary_dimension": None,
                "reasoning": "Not specified"
            },
            "relationships": [],
            "data_quality_insights": {
                "completeness": "Not assessed",
                "consistency": "Not assessed",
                "outliers": "Not assessed",
                "data_issues": []
            },
            "analysis_recommendations": [],
            "advanced_insights": {
                "seasonality_potential": "Unknown",
                "segmentation_opportunities": [],
                "kpi_suggestions": [],
                "dashboard_recommendations": []
            }
        }
        
        # Deep merge with defaults
        def deep_merge(default: Dict, actual: Dict) -> Dict:
            """Recursively merge actual values into default structure."""
            result = default.copy()
            for key, value in actual.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(defaults, analysis)
    
    def _assess_data_quality_with_ai(self, df: pd.DataFrame, gemini_analysis: Dict) -> Dict[str, Any]:
        """Combine AI insights with statistical data quality assessment."""
        issues = []
        
        # Use Gemini's quality insights
        gemini_quality = gemini_analysis.get('data_quality_insights', {})
        
        # Add Gemini-identified issues
        for issue in gemini_quality.get('data_issues', []):
            issues.append({
                "type": "ai_detected",
                "severity": "medium",
                "message": issue,
                "source": "gemini_analysis"
            })
        
        # Statistical checks
        # Check for completely null columns
        null_columns = df.columns[df.isna().all()].tolist()
        if null_columns:
            issues.append({
                "type": "null_columns",
                "severity": "high",
                "columns": null_columns,
                "message": f"Found {len(null_columns)} completely null columns"
            })
        
        # Calculate overall quality score
        quality_score = max(0, 100 - len(issues) * 10)
        
        return {
            "overall_quality_score": quality_score,
            "total_issues": len(issues),
            "issues": issues,
            "ai_assessment": gemini_quality.get('completeness', 'Not assessed'),
            "consistency_notes": gemini_quality.get('consistency', 'Not assessed'),
            "outlier_summary": gemini_quality.get('outliers', 'Not assessed')
        }


# Convenience functions that match the original API

def profile_dataset_with_gemini(df: pd.DataFrame, dataset_name: str, api_key: str = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Profile a dataset using Gemini AI.
    
    Args:
        df: DataFrame to profile
        dataset_name: Name of the dataset
        api_key: Google API key (optional, will check environment)
        use_cache: Whether to use cached configurations if available
        
    Returns:
        Complete profiling results
    """
    profiler = IntelligentDatasetProfiler(api_key)
    return profiler.profile_with_gemini(df, dataset_name, use_cache)


def generate_config_from_gemini_profile(df: pd.DataFrame, dataset_name: str, api_key: str = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Generate a configuration using Gemini AI profiling.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
        api_key: Google API key (optional, will check environment)
        use_cache: Whether to use cached configurations if available
        
    Returns:
        Configuration dictionary
    """
    profiler = IntelligentDatasetProfiler(api_key)
    profile = profiler.profile_with_gemini(df, dataset_name, use_cache)
    return profile.get('configuration') if profile else None 