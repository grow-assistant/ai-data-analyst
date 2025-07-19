import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)



def calculate_dynamic_metrics(df: pd.DataFrame, metrics_config: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculates metrics dynamically based on the configuration.
    
    Args:
        df: DataFrame with the source data
        metrics_config: List of metric configurations from config.yaml
    
    Returns:
        DataFrame with calculated metrics added as new columns
    """
    df_copy = df.copy()
    
    for metric_config in metrics_config:
        metric_name = metric_config.get('name')
        numerator_col = metric_config.get('numerator')
        denominator_col = metric_config.get('denominator')
        aggregation_method = metric_config.get('aggregation_method', 'ratio')
        
        if not metric_name:
            logger.warning(f"Metric configuration missing 'name': {metric_config}")
            continue
            
        try:
            if aggregation_method == 'ratio':
                if not numerator_col or not denominator_col:
                    logger.warning(f"Ratio metric '{metric_name}' missing numerator or denominator columns")
                    df_copy[metric_name] = 0
                    continue
                    
                if numerator_col not in df_copy.columns:
                    logger.warning(f"Numerator column '{numerator_col}' not found for metric '{metric_name}'")
                    df_copy[metric_name] = 0
                    continue
                    
                if denominator_col not in df_copy.columns:
                    logger.warning(f"Denominator column '{denominator_col}' not found for metric '{metric_name}'")
                    df_copy[metric_name] = 0
                    continue
                
                # Calculate ratio metric
                denominator_values = df_copy[denominator_col]
                numerator_values = df_copy[numerator_col]
                
                # Handle division by zero
                df_copy[metric_name] = np.where(
                    denominator_values != 0,
                    numerator_values / denominator_values,
                    0
                )
                
                # Handle NaN values
                df_copy[metric_name] = df_copy[metric_name].fillna(0)
                
                logger.info(f"Calculated ratio metric '{metric_name}' = {numerator_col} / {denominator_col}")
                
            elif aggregation_method == 'sum':
                if not numerator_col:
                    logger.warning(f"Sum metric '{metric_name}' missing numerator column")
                    df_copy[metric_name] = 0
                    continue
                    
                if numerator_col not in df_copy.columns:
                    logger.warning(f"Numerator column '{numerator_col}' not found for metric '{metric_name}'")
                    df_copy[metric_name] = 0
                    continue
                
                # For sum metrics, just use the column as-is (it will be aggregated later)
                df_copy[metric_name] = df_copy[numerator_col]
                logger.info(f"Set up sum metric '{metric_name}' = {numerator_col}")
                
            else:
                logger.warning(f"Unknown aggregation method '{aggregation_method}' for metric '{metric_name}'")
                df_copy[metric_name] = 0
                
        except Exception as e:
            logger.error(f"Error calculating metric '{metric_name}': {e}")
            df_copy[metric_name] = 0
    
    return df_copy 


def calculate_metrics(data: str, metrics_config: str) -> str:
    """
    ADK tool wrapper for calculating dynamic metrics.
    
    Args:
        data: CSV string containing the dataset
        metrics_config: JSON string containing metric configurations
    
    Returns:
        String summary of the calculated metrics
    """
    try:
        import io
        import json
        
        df = pd.read_csv(io.StringIO(data))
        
        # Parse metrics configuration
        try:
            config = json.loads(metrics_config)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in metrics_config parameter"
        
        # Calculate metrics
        result_df = calculate_dynamic_metrics(df, config)
        
        result_summary = f"Dynamic Metrics Calculation Results\n"
        result_summary += f"Original columns: {len(df.columns)}\n"
        result_summary += f"Columns after metrics: {len(result_df.columns)}\n"
        
        # Show new metric columns
        new_cols = [col for col in result_df.columns if col not in df.columns]
        if new_cols:
            result_summary += f"New metric columns: {', '.join(new_cols)}\n"
            
            # Show sample values for new metrics
            for col in new_cols[:5]:  # Show first 5 metrics
                if col in result_df.columns:
                    mean_val = result_df[col].mean()
                    result_summary += f"  {col}: mean = {mean_val:.4f}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in metrics calculation: {str(e)}" 