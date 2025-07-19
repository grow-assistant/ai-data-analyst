"""
Schema Profiler Agent Executor
This module contains the implementation of the Schema Profiler Agent's skills.
"""

import logging
import pandas as pd
from typing import Dict, Any, List
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

# Import the intelligent profiler
try:
    from .intelligent_profiler import IntelligentDatasetProfiler, profile_dataset_with_gemini
    INTELLIGENT_PROFILER_AVAILABLE = True
except ImportError:
    INTELLIGENT_PROFILER_AVAILABLE = False
    print("⚠️ Intelligent profiler not available. Using basic profiling only.")

logger = logging.getLogger(__name__)

class SchemaProfilerAgentExecutor:
    """
    Implements the logic for the schema profiler agent's skills.
    """
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        logger.info("SchemaProfilerAgentExecutor initialized.")

    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column and return its profile."""
        column_profile = {
            "name": series.name,
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": int(series.isnull().sum()),
            "null_percentage": float(series.isnull().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "duplicate_count": int(len(series) - series.nunique()),
        }

        # Add type-specific analysis
        if pd.api.types.is_numeric_dtype(series):
            column_profile.update({
                "data_type": "numeric",
                "min_value": float(series.min()) if not series.empty else None,
                "max_value": float(series.max()) if not series.empty else None,
                "mean": float(series.mean()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
                "std_dev": float(series.std()) if not series.empty else None,
                "quartiles": {
                    "q1": float(series.quantile(0.25)) if not series.empty else None,
                    "q3": float(series.quantile(0.75)) if not series.empty else None
                }
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_profile.update({
                "data_type": "datetime",
                "min_date": str(series.min()) if not series.empty else None,
                "max_date": str(series.max()) if not series.empty else None,
                "date_range_days": (series.max() - series.min()).days if not series.empty else None
            })
        else:
            column_profile.update({
                "data_type": "categorical",
                "most_frequent": str(series.mode().iloc[0]) if not series.empty and len(series.mode()) > 0 else None,
                "avg_length": float(series.astype(str).str.len().mean()) if not series.empty else None,
                "max_length": int(series.astype(str).str.len().max()) if not series.empty else None,
                "min_length": int(series.astype(str).str.len().min()) if not series.empty else None
            })

        # Sample values (top 5 most frequent)
        if not series.empty:
            value_counts = series.value_counts().head(5)
            column_profile["sample_values"] = [
                {"value": str(val), "count": int(count)} 
                for val, count in value_counts.items()
            ]
        else:
            column_profile["sample_values"] = []

        return column_profile

    def _generate_schema_suggestions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate schema suggestions based on data analysis."""
        suggestions = {
            "primary_key_candidates": [],
            "foreign_key_candidates": [],
            "indexing_suggestions": [],
            "data_quality_issues": []
        }

        for col in df.columns:
            # Check for primary key candidates (unique, non-null)
            if df[col].nunique() == len(df) and df[col].isnull().sum() == 0:
                suggestions["primary_key_candidates"].append(col)

            # Check for columns that need indexing (high cardinality, frequently queried)
            if df[col].nunique() > len(df) * 0.1 and df[col].nunique() < len(df) * 0.9:
                suggestions["indexing_suggestions"].append({
                    "column": col,
                    "reason": "High selectivity column"
                })

            # Data quality issues
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > 50:
                suggestions["data_quality_issues"].append({
                    "column": col,
                    "issue": f"High null percentage: {null_pct:.1f}%"
                })

        return suggestions

    async def ai_profile_dataset_skill(self, data_handle_id: str, use_cache: bool = True, force_ai: bool = False) -> Dict[str, Any]:
        """
        A2A skill to profile a dataset using AI-powered analysis with configuration caching.
        """
        logger.info(f"Executing ai_profile_dataset_skill for data handle: {data_handle_id}")

        if not INTELLIGENT_PROFILER_AVAILABLE and force_ai:
            raise ValueError("Intelligent profiler not available. Install google-generativeai package.")

        try:
            # Get data from handle
            handle = self.data_manager.get_handle(data_handle_id)
            if not handle:
                raise ValueError(f"Data handle not found: {data_handle_id}")

            df = self.data_manager.get_data(data_handle_id)
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Data handle does not contain a DataFrame: {type(df)}")

            logger.info(f"AI Profiling DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Get dataset name from handle metadata or use handle ID
            dataset_name = handle.metadata.get('original_filename', data_handle_id)
            if dataset_name.endswith(('.csv', '.tdsx', '.json', '.xlsx')):
                dataset_name = dataset_name.rsplit('.', 1)[0]

            # Use intelligent profiler if available
            if INTELLIGENT_PROFILER_AVAILABLE:
                try:
                    ai_profile = profile_dataset_with_gemini(df, dataset_name, use_cache=use_cache)
                    
                    if ai_profile:
                        # Create result data handle for the AI profile
                        profile_handle = self.data_manager.create_handle(
                            data=ai_profile,
                            data_type="ai_schema_profile",
                            metadata={
                                "source_data_handle": data_handle_id,
                                "dataset_name": dataset_name,
                                "profiled_at": datetime.now().isoformat(),
                                "dataset_rows": len(df),
                                "dataset_columns": len(df.columns),
                                "profile_type": "ai_powered",
                                "cached": ai_profile.get('cached', False)
                            }
                        )

                        logger.info(f"Successfully AI profiled dataset and created profile handle {profile_handle.handle_id}")

                        return {
                            "status": "completed",
                            "profile_handle_id": profile_handle.handle_id,
                            "profile_data": ai_profile,
                            "configuration": ai_profile.get('configuration'),
                            "message": f"Successfully AI profiled dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns.",
                            "summary": {
                                "dataset_type": ai_profile.get('gemini_analysis', {}).get('dataset_summary', {}).get('dataset_type', 'Unknown'),
                                "domain": ai_profile.get('gemini_analysis', {}).get('dataset_summary', {}).get('domain', 'Unknown'),
                                "total_columns": len(df.columns),
                                "total_rows": len(df),
                                "dimensions_found": len(ai_profile.get('dimensions', [])),
                                "measures_found": len(ai_profile.get('measures', [])),
                                "recommendations": len(ai_profile.get('analysis_recommendations', [])),
                                "cached": ai_profile.get('cached', False)
                            }
                        }
                    else:
                        # Fall back to basic profiling if AI fails
                        logger.warning("AI profiling failed, falling back to basic profiling")
                        return await self.profile_dataset_skill(data_handle_id, "comprehensive")
                        
                except Exception as ai_error:
                    logger.warning(f"AI profiling failed: {ai_error}")
                    if force_ai:
                        raise
                    # Fall back to basic profiling
                    return await self.profile_dataset_skill(data_handle_id, "comprehensive")
            else:
                # Fall back to basic profiling
                logger.info("Intelligent profiler not available, using basic profiling")
                return await self.profile_dataset_skill(data_handle_id, "comprehensive")

        except Exception as e:
            logger.exception(f"Error in AI profiling dataset: {e}")
            raise

    async def profile_dataset_skill(self, data_handle_id: str, profile_type: str = "comprehensive") -> Dict[str, Any]:
        """
        A2A skill to profile a dataset from a data handle.
        """
        logger.info(f"Executing profile_dataset_skill for data handle: {data_handle_id}")

        try:
            # Get data from handle using the correct method
            handle = self.data_manager.get_handle(data_handle_id)
            if not handle:
                raise ValueError(f"Data handle not found: {data_handle_id}")

            # Get the actual data using get_data method
            df = self.data_manager.get_data(data_handle_id)
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Data handle does not contain a DataFrame: {type(df)}")

            logger.info(f"Profiling DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Basic dataset statistics
            dataset_profile = {
                "dataset_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
                    "column_names": list(df.columns),
                    "duplicated_rows": int(df.duplicated().sum()),
                    "completely_empty_rows": int((df.isnull().all(axis=1)).sum())
                },
                "columns": [],
                "data_types_summary": {},
                "data_quality_score": 0.0
            }

            # Analyze each column
            for column in df.columns:
                try:
                    column_profile = self._analyze_column(df[column])
                    dataset_profile["columns"].append(column_profile)
                except Exception as e:
                    logger.warning(f"Failed to analyze column {column}: {e}")
                    dataset_profile["columns"].append({
                        "name": column,
                        "error": str(e),
                        "dtype": str(df[column].dtype)
                    })

            # Data types summary
            dtype_counts = df.dtypes.value_counts()
            dataset_profile["data_types_summary"] = {
                str(dtype): int(count) for dtype, count in dtype_counts.items()
            }

            # Generate schema suggestions if comprehensive profiling
            if profile_type == "comprehensive":
                dataset_profile["schema_suggestions"] = self._generate_schema_suggestions(df)

            # Calculate data quality score (0-100)
            total_cells = len(df) * len(df.columns)
            non_null_cells = total_cells - df.isnull().sum().sum()
            unique_rows = len(df) - df.duplicated().sum()
            
            quality_score = (
                (non_null_cells / total_cells * 0.4) +  # 40% for completeness
                (unique_rows / len(df) * 0.3) +         # 30% for uniqueness
                (len([col for col in df.columns if df[col].nunique() > 1]) / len(df.columns) * 0.3)  # 30% for variety
            ) * 100

            dataset_profile["data_quality_score"] = round(quality_score, 2)

            # Create result data handle for the profile
            profile_handle = self.data_manager.create_handle(
                data=dataset_profile,
                data_type="schema_profile",
                metadata={
                    "source_data_handle": data_handle_id,
                    "profile_type": profile_type,
                    "profiled_at": datetime.now().isoformat(),
                    "dataset_rows": len(df),
                    "dataset_columns": len(df.columns)
                }
            )

            logger.info(f"Successfully profiled dataset and created profile handle {profile_handle.handle_id}")

            return {
                "status": "completed",
                "profile_handle_id": profile_handle.handle_id,
                "profile_data": dataset_profile,
                "message": f"Successfully profiled dataset with {len(df)} rows and {len(df.columns)} columns.",
                "summary": {
                    "data_quality_score": dataset_profile["data_quality_score"],
                    "total_columns": len(df.columns),
                    "total_rows": len(df),
                    "data_types": len(dataset_profile["data_types_summary"])
                }
            }

        except Exception as e:
            logger.exception(f"Error profiling dataset: {e}")
            raise

    async def get_column_statistics_skill(self, data_handle_id: str, column_name: str) -> Dict[str, Any]:
        """
        A2A skill to get detailed statistics for a specific column.
        """
        logger.info(f"Executing get_column_statistics_skill for column: {column_name}")

        try:
            handle = self.data_manager.get_handle(data_handle_id)
            if not handle:
                raise ValueError(f"Data handle not found: {data_handle_id}")

            df = self.data_manager.get_data(data_handle_id)
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")

            column_stats = self._analyze_column(df[column_name])
            
            return {
                "status": "completed",
                "column_statistics": column_stats,
                "message": f"Successfully analyzed column '{column_name}'"
            }

        except Exception as e:
            logger.exception(f"Error analyzing column {column_name}: {e}")
            raise

    async def compare_schemas_skill(self, data_handle_id1: str, data_handle_id2: str) -> Dict[str, Any]:
        """
        A2A skill to compare schemas between two datasets.
        """
        logger.info(f"Comparing schemas between handles: {data_handle_id1} and {data_handle_id2}")

        try:
            handle1 = self.data_manager.get_handle(data_handle_id1)
            handle2 = self.data_manager.get_handle(data_handle_id2)
            
            if not handle1 or not handle2:
                raise ValueError("One or both data handles not found")

            df1 = self.data_manager.get_data(data_handle_id1)
            df2 = self.data_manager.get_data(data_handle_id2)

            comparison = {
                "common_columns": list(set(df1.columns) & set(df2.columns)),
                "unique_to_first": list(set(df1.columns) - set(df2.columns)),
                "unique_to_second": list(set(df2.columns) - set(df1.columns)),
                "schema_compatibility": len(set(df1.columns) & set(df2.columns)) / len(set(df1.columns) | set(df2.columns)),
                "type_mismatches": []
            }

            # Check for type mismatches in common columns
            for col in comparison["common_columns"]:
                if str(df1[col].dtype) != str(df2[col].dtype):
                    comparison["type_mismatches"].append({
                        "column": col,
                        "first_type": str(df1[col].dtype),
                        "second_type": str(df2[col].dtype)
                    })

            return {
                "status": "completed",
                "comparison": comparison,
                "message": f"Schema comparison completed for {len(comparison['common_columns'])} common columns"
            }

        except Exception as e:
            logger.exception(f"Error comparing schemas: {e}")
            raise

    async def get_dataset_config_skill(self, data_handle_id: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        A2A skill to get cached configuration for a dataset.
        """
        logger.info(f"Checking for cached configuration for dataset: {dataset_name or data_handle_id}")

        try:
            if not INTELLIGENT_PROFILER_AVAILABLE:
                return {
                    "status": "completed",
                    "cached_config": None,
                    "message": "Intelligent profiler not available - no cached configurations"
                }

            # Get dataset name if not provided
            if not dataset_name:
                handle = self.data_manager.get_handle(data_handle_id)
                if handle:
                    dataset_name = handle.metadata.get('original_filename', data_handle_id)
                    if dataset_name.endswith(('.csv', '.tdsx', '.json', '.xlsx')):
                        dataset_name = dataset_name.rsplit('.', 1)[0]
                else:
                    dataset_name = data_handle_id

            # Check for cached configuration
            profiler = IntelligentDatasetProfiler()
            
            # Try to get the dataset to compute hash
            try:
                df = self.data_manager.get_data(data_handle_id)
                dataset_hash = profiler.generate_dataset_hash(df, dataset_name)
                cached_config = profiler.get_cached_config(dataset_name, dataset_hash)
            except Exception:
                # If we can't get the data, just check by name without hash validation
                cached_config = profiler.get_cached_config(dataset_name, "")

            if cached_config:
                return {
                    "status": "completed",
                    "cached_config": cached_config,
                    "config_date": cached_config.get('configuration_date'),
                    "config_source": cached_config.get('configuration_source'),
                    "message": f"Found cached configuration for {dataset_name}",
                    "summary": {
                        "dataset_type": cached_config.get('ai_insights', {}).get('dataset_type', 'Unknown'),
                        "domain": cached_config.get('ai_insights', {}).get('domain', 'Unknown'),
                        "total_metrics": len(cached_config.get('column_mappings', {}).get('all_metrics', [])),
                        "total_dimensions": len(cached_config.get('column_mappings', {}).get('all_dimensions', [])),
                        "has_date_columns": len(cached_config.get('column_mappings', {}).get('all_dates', [])) > 0,
                        "recommendations": len(cached_config.get('ai_insights', {}).get('recommendations', []))
                    }
                }
            else:
                return {
                    "status": "completed", 
                    "cached_config": None,
                    "message": f"No cached configuration found for {dataset_name}"
                }

        except Exception as e:
            logger.exception(f"Error checking cached configuration: {e}")
            raise

    async def list_all_configs_skill(self) -> Dict[str, Any]:
        """
        A2A skill to list all cached dataset configurations.
        """
        logger.info("Listing all cached dataset configurations")

        try:
            if not INTELLIGENT_PROFILER_AVAILABLE:
                return {
                    "status": "completed",
                    "configs": [],
                    "message": "Intelligent profiler not available"
                }

            profiler = IntelligentDatasetProfiler()
            config_files = list(profiler.cache_dir.glob("config_*.json"))
            
            configs = []
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    configs.append({
                        "dataset_name": config.get('dataset_name'),
                        "config_file": str(config_file),
                        "configuration_date": config.get('configuration_date'),
                        "dataset_type": config.get('ai_insights', {}).get('dataset_type'),
                        "domain": config.get('ai_insights', {}).get('domain'),
                        "metrics_count": len(config.get('column_mappings', {}).get('all_metrics', [])),
                        "dimensions_count": len(config.get('column_mappings', {}).get('all_dimensions', []))
                    })
                except Exception as e:
                    logger.warning(f"Error reading config file {config_file}: {e}")

            return {
                "status": "completed",
                "configs": configs,
                "total_configs": len(configs),
                "message": f"Found {len(configs)} cached configurations"
            }

        except Exception as e:
            logger.exception(f"Error listing configurations: {e}")
            raise
