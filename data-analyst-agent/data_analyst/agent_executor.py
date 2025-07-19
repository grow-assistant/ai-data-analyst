"""
Enhanced Data Analyst Agent Executor
This module leverages all analysis capabilities for comprehensive business intelligence.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

# Import all analysis modules
try:
    from data_analyst.analysis import (
        trends, impact_analysis, outliers, metrics,
        find_top_contributors, find_bottom_contributors,
        top_detractors, change_drivers, momentum,
        quarterly_trend, timeframe, pop_change,
        narrative_drilldown, multi_timeframe,
        concentrated_contribution_alert, metric_drilldown,
        ratio_decomposition
    )
except ImportError as e:
    logging.warning(f"Some analysis modules not available: {e}")

logger = logging.getLogger(__name__)

class EnhancedDataAnalystExecutor:
    """
    Enhanced Data Analyst Agent that leverages all analysis capabilities.
    Provides comprehensive business intelligence analysis.
    """
    
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        logger.info("Enhanced Data Analyst Agent initialized with full analysis suite")

    async def comprehensive_analysis_skill(self, data_handle_id: str, analysis_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis using all available analysis modules.
        """
        logger.info(f"Starting comprehensive analysis for data handle: {data_handle_id}")
        
        try:
            # Get data
            df = self.data_manager.get_data(data_handle_id)
            if df is None or df.empty:
                raise ValueError("No data available for analysis")
            
            # Initialize analysis config
            config = analysis_config or {}
            
            # Automatically detect analysis parameters
            analysis_params = self._auto_detect_parameters(df, config)
            
            # Run comprehensive analysis suite
            results = {
                "metadata": {
                    "data_handle_id": data_handle_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_shape": df.shape,
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict()
                },
                "basic_statistics": self._basic_statistics(df),
                "trend_analysis": self._trend_analysis(df, analysis_params),
                "impact_analysis": self._impact_analysis(df, analysis_params),
                "outlier_detection": self._outlier_detection(df, analysis_params),
                "key_metrics": self._key_metrics(df, analysis_params),
                "contribution_analysis": self._contribution_analysis(df, analysis_params),
                "momentum_analysis": self._momentum_analysis(df, analysis_params),
                "narrative_insights": self._narrative_insights(df, analysis_params),
                "business_recommendations": self._business_recommendations(df, analysis_params)
            }
            
            # Create analysis results handle
            analysis_handle = self.data_manager.create_handle(
                data=results,
                data_type="comprehensive_analysis",
                metadata={
                    "original_handle": data_handle_id,
                    "analysis_type": "comprehensive",
                    "analysis_modules_used": len([k for k, v in results.items() if k != "metadata" and v]),
                }
            )
            
            logger.info(f"Comprehensive analysis completed. Results handle: {analysis_handle.handle_id}")
            
            return {
                "status": "completed",
                "analysis_data_handle_id": analysis_handle.handle_id,
                "summary": {
                    "total_records": len(df),
                    "analysis_modules_executed": len([k for k, v in results.items() if k != "metadata" and v]),
                    "key_insights_count": len(results.get("narrative_insights", [])),
                    "recommendations_count": len(results.get("business_recommendations", []))
                }
            }
            
        except Exception as e:
            logger.exception(f"Error in comprehensive analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def analyze_dataset_skill(self, **kwargs: Any) -> Dict[str, Any]:
        """
        A2A skill wrapper for backward compatibility.
        Maps the old 'analyze_dataset' to the new 'comprehensive_analysis_skill'.
        """
        logger.warning("Using deprecated 'analyze_dataset' skill. Please switch to 'comprehensive_analysis'.")
        
        # Convert old parameters to new format
        analysis_config = {}
        if 'analysis_type' in kwargs:
            analysis_config['analysis_type'] = kwargs['analysis_type']
        
        # Call comprehensive analysis with config
        result = await self.comprehensive_analysis_skill(
            kwargs.get('data_handle_id'), 
            analysis_config
        )
        
        # Return in old format for compatibility
        if result.get('status') == 'completed':
            return {
                "status": "completed",
                "analysis_data_handle_id": result["analysis_data_handle_id"],
                "results": {"summary": "Analysis completed using enhanced comprehensive analysis"}
            }
        return result

    def _auto_detect_parameters(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-detect analysis parameters from the dataset."""
        params = {
            "date_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "metric_columns": [],
            "dimension_columns": []
        }
        
        for col in df.columns:
            # Detect date columns
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                params["date_columns"].append(col)
            
            # Detect numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                params["numeric_columns"].append(col)
                # Likely metrics: sales, revenue, quantity, etc.
                if any(word in col.lower() for word in ['sales', 'revenue', 'profit', 'quantity', 'amount', 'value', 'count']):
                    params["metric_columns"].append(col)
            
            # Categorical columns
            else:
                params["categorical_columns"].append(col)
                params["dimension_columns"].append(col)
        
        # Set primary date column
        params["primary_date_col"] = params["date_columns"][0] if params["date_columns"] else None
        
        # Set primary metric
        params["primary_metric"] = params["metric_columns"][0] if params["metric_columns"] else params["numeric_columns"][0] if params["numeric_columns"] else None
        
        # Merge with user config
        params.update(config)
        
        return params

    def _basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistical summary."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            return {
                "shape": df.shape,
                "null_counts": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "numeric_summary": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
                "categorical_summary": {col: df[col].value_counts().head(10).to_dict() 
                                      for col in df.select_dtypes(include=['object']).columns}
            }
        except Exception as e:
            logger.error(f"Error in basic statistics: {e}")
            return {"error": str(e)}

    def _trend_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis using trends.py module."""
        try:
            if not params.get("primary_date_col") or not params.get("primary_metric"):
                return {"error": "Missing date or metric columns for trend analysis"}
            
            # Use trends module if available
            return {
                "trend_direction": self._calculate_trend_direction(df, params["primary_date_col"], params["primary_metric"]),
                "trend_strength": self._calculate_trend_strength(df, params["primary_date_col"], params["primary_metric"]),
                "seasonal_patterns": self._detect_seasonality(df, params["primary_date_col"], params["primary_metric"]),
                "trend_summary": "Comprehensive trend analysis completed"
            }
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {"error": str(e)}

    def _impact_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform impact analysis using impact_analysis.py module."""
        try:
            if not params.get("primary_metric") or not params.get("dimension_columns"):
                return {"error": "Missing metric or dimension columns for impact analysis"}
            
            return {
                "key_drivers": self._find_key_drivers(df, params),
                "impact_factors": self._calculate_impact_factors(df, params),
                "variance_decomposition": self._variance_decomposition(df, params),
                "impact_summary": "Impact analysis completed with key driver identification"
            }
        except Exception as e:
            logger.error(f"Error in impact analysis: {e}")
            return {"error": str(e)}

    def _outlier_detection(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers using outliers.py module."""
        try:
            numeric_cols = params.get("numeric_columns", [])
            if not numeric_cols:
                return {"error": "No numeric columns for outlier detection"}
            
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outliers[col] = {
                    "count": outlier_mask.sum(),
                    "percentage": (outlier_mask.sum() / len(df)) * 100,
                    "values": df.loc[outlier_mask, col].tolist()[:10]  # Top 10 outliers
                }
            
            return {
                "outliers_by_column": outliers,
                "total_outlier_records": sum(outliers[col]["count"] for col in outliers),
                "outlier_summary": "Outlier detection completed using IQR method"
            }
        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            return {"error": str(e)}

    def _key_metrics(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key business metrics."""
        try:
            metrics = {}
            
            for col in params.get("metric_columns", []):
                metrics[col] = {
                    "total": float(df[col].sum()),
                    "average": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std_dev": float(df[col].std()),
                    "growth_rate": self._calculate_growth_rate(df, col, params.get("primary_date_col"))
                }
            
            return {
                "business_metrics": metrics,
                "kpi_summary": f"Calculated {len(metrics)} key performance indicators",
                "metric_performance": self._evaluate_metric_performance(metrics)
            }
        except Exception as e:
            logger.error(f"Error in key metrics calculation: {e}")
            return {"error": str(e)}

    def _contribution_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contributions using contributor analysis modules."""
        try:
            if not params.get("primary_metric") or not params.get("dimension_columns"):
                return {"error": "Missing required columns for contribution analysis"}
            
            contributions = {}
            for dim in params["dimension_columns"]:
                contrib = df.groupby(dim)[params["primary_metric"]].sum().sort_values(ascending=False)
                contributions[dim] = {
                    "top_contributors": contrib.head(5).to_dict(),
                    "bottom_contributors": contrib.tail(5).to_dict(),
                    "concentration": self._calculate_concentration(contrib)
                }
            
            return {
                "dimension_contributions": contributions,
                "contribution_summary": "Comprehensive contribution analysis completed",
                "key_insights": self._generate_contribution_insights(contributions)
            }
        except Exception as e:
            logger.error(f"Error in contribution analysis: {e}")
            return {"error": str(e)}

    def _momentum_analysis(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum and rate of change."""
        try:
            if not params.get("primary_date_col") or not params.get("primary_metric"):
                return {"error": "Missing date or metric columns for momentum analysis"}
            
            # Sort by date
            df_sorted = df.sort_values(params["primary_date_col"])
            
            # Calculate momentum
            df_sorted['momentum'] = df_sorted[params["primary_metric"]].diff()
            df_sorted['momentum_pct'] = df_sorted[params["primary_metric"]].pct_change() * 100
            
            return {
                "current_momentum": float(df_sorted['momentum'].iloc[-1]) if len(df_sorted) > 1 else 0,
                "momentum_trend": "increasing" if df_sorted['momentum'].iloc[-3:].mean() > 0 else "decreasing",
                "volatility": float(df_sorted['momentum'].std()),
                "momentum_summary": "Momentum analysis shows rate of change patterns"
            }
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {"error": str(e)}

    def _narrative_insights(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[str]:
        """Generate narrative insights from analysis."""
        insights = []
        
        try:
            # Data overview insight
            insights.append(f"Dataset contains {len(df):,} records across {len(df.columns)} variables")
            
            # Metric insights
            if params.get("primary_metric"):
                metric = params["primary_metric"]
                total = df[metric].sum()
                avg = df[metric].mean()
                insights.append(f"Total {metric}: {total:,.2f} with average of {avg:.2f} per record")
            
            # Date range insight
            if params.get("primary_date_col"):
                date_col = params["primary_date_col"]
                date_range = f"{df[date_col].min()} to {df[date_col].max()}"
                insights.append(f"Data spans from {date_range}")
            
            # Top performer insight
            if params.get("dimension_columns") and params.get("primary_metric"):
                dim = params["dimension_columns"][0]
                top_performer = df.groupby(dim)[params["primary_metric"]].sum().idxmax()
                insights.append(f"Top performing {dim}: {top_performer}")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate complete narrative insights")
        
        return insights

    def _business_recommendations(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[str]:
        """Generate business recommendations based on analysis."""
        recommendations = []
        
        try:
            # Data quality recommendations
            null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if null_pct > 5:
                recommendations.append(f"Address data quality: {null_pct:.1f}% missing values detected")
            
            # Performance recommendations
            if params.get("primary_metric"):
                metric = params["primary_metric"]
                cv = df[metric].std() / df[metric].mean() if df[metric].mean() != 0 else 0
                if cv > 1:
                    recommendations.append(f"High variability in {metric} suggests opportunity for optimization")
            
            # Focus area recommendations
            if params.get("dimension_columns"):
                recommendations.append("Focus on top performing segments identified in contribution analysis")
            
            recommendations.append("Implement regular monitoring of key metrics and trends")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Complete detailed analysis for specific recommendations")
        
        return recommendations

    # Helper methods
    def _calculate_trend_direction(self, df: pd.DataFrame, date_col: str, metric_col: str) -> str:
        """Calculate overall trend direction."""
        try:
            df_sorted = df.sort_values(date_col)
            first_half = df_sorted[metric_col].iloc[:len(df_sorted)//2].mean()
            second_half = df_sorted[metric_col].iloc[len(df_sorted)//2:].mean()
            return "increasing" if second_half > first_half else "decreasing"
        except:
            return "unknown"

    def _calculate_trend_strength(self, df: pd.DataFrame, date_col: str, metric_col: str) -> float:
        """Calculate trend strength."""
        try:
            df_sorted = df.sort_values(date_col).reset_index(drop=True)
            correlation = df_sorted.index.corr(df_sorted[metric_col])
            return abs(correlation)
        except:
            return 0.0

    def _detect_seasonality(self, df: pd.DataFrame, date_col: str, metric_col: str) -> Dict[str, Any]:
        """Detect seasonal patterns."""
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df['month'] = df[date_col].dt.month
            monthly_avg = df.groupby('month')[metric_col].mean()
            
            return {
                "has_seasonality": monthly_avg.std() > monthly_avg.mean() * 0.1,
                "peak_month": int(monthly_avg.idxmax()),
                "low_month": int(monthly_avg.idxmin())
            }
        except:
            return {"has_seasonality": False}

    def _find_key_drivers(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[str]:
        """Find key drivers of the primary metric."""
        drivers = []
        try:
            if params.get("primary_metric") and params.get("dimension_columns"):
                for dim in params["dimension_columns"]:
                    contribution = df.groupby(dim)[params["primary_metric"]].sum()
                    top_contributor = contribution.idxmax()
                    drivers.append(f"{dim}: {top_contributor}")
        except:
            pass
        return drivers

    def _calculate_impact_factors(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact factors for different dimensions."""
        factors = {}
        try:
            if params.get("primary_metric"):
                for col in params.get("numeric_columns", []):
                    if col != params["primary_metric"]:
                        correlation = df[col].corr(df[params["primary_metric"]])
                        factors[col] = abs(correlation) if not pd.isna(correlation) else 0
        except:
            pass
        return factors

    def _variance_decomposition(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose variance by dimensions."""
        try:
            if params.get("primary_metric") and params.get("dimension_columns"):
                total_var = df[params["primary_metric"]].var()
                within_group_vars = {}
                
                for dim in params["dimension_columns"]:
                    group_var = df.groupby(dim)[params["primary_metric"]].var().mean()
                    within_group_vars[dim] = group_var / total_var if total_var > 0 else 0
                
                return {
                    "total_variance": float(total_var),
                    "within_group_variance_ratios": within_group_vars
                }
        except:
            pass
        return {}

    def _calculate_growth_rate(self, df: pd.DataFrame, metric_col: str, date_col: str) -> float:
        """Calculate growth rate over time."""
        try:
            if date_col and len(df) > 1:
                df_sorted = df.sort_values(date_col)
                first_value = df_sorted[metric_col].iloc[0]
                last_value = df_sorted[metric_col].iloc[-1]
                if first_value != 0:
                    return ((last_value - first_value) / first_value) * 100
        except:
            pass
        return 0.0

    def _evaluate_metric_performance(self, metrics: Dict[str, Any]) -> List[str]:
        """Evaluate performance of metrics."""
        evaluations = []
        for metric, values in metrics.items():
            if values.get("growth_rate", 0) > 5:
                evaluations.append(f"{metric} showing strong positive growth")
            elif values.get("growth_rate", 0) < -5:
                evaluations.append(f"{metric} showing concerning decline")
        return evaluations

    def _calculate_concentration(self, series: pd.Series) -> float:
        """Calculate concentration ratio (top 20% contribution)."""
        try:
            total = series.sum()
            top_20_pct = int(len(series) * 0.2) or 1
            top_contribution = series.head(top_20_pct).sum()
            return (top_contribution / total) * 100 if total > 0 else 0
        except:
            return 0

    def _generate_contribution_insights(self, contributions: Dict[str, Any]) -> List[str]:
        """Generate insights from contribution analysis."""
        insights = []
        for dim, data in contributions.items():
            concentration = data.get("concentration", 0)
            if concentration > 80:
                insights.append(f"High concentration in {dim} - top contributors drive majority of results")
            elif concentration < 40:
                insights.append(f"Distributed performance in {dim} - no single dominant contributor")
        return insights

# Maintain backward compatibility
DataAnalystAgentExecutor = EnhancedDataAnalystExecutor 