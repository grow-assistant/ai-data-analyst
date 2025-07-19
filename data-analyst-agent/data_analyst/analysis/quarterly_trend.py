# data_analyzer/analysis/quarterly_trend.py
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from .change_drivers import _calculate_metric
from .timeframe import get_reference_date_from_config

logger = logging.getLogger(__name__)

def calculate_quarterly_trend(df: pd.DataFrame, metrics: List[Dict[str, Any]], date_col: str, config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Calculates the Compound Weekly Growth Rate (CWGR) over the full dataset period (assumed to be ~13 weeks).
    """
    trend_config = config.get('analysis', {}).get('quarterly_trend_analysis', {})
    if not trend_config.get('enabled', False):
        return None

    logger.info("--- Calculating Quarterly Trend (CWGR) ---")

    reference_date = get_reference_date_from_config(config, df, date_col)
    
    first_week_start = df[date_col].min()
    first_week_end = first_week_start + pd.DateOffset(days=6)
    
    last_week_end = reference_date
    last_week_start = last_week_end - pd.DateOffset(days=6)
    
    num_weeks = (last_week_end - first_week_start).days / 7
    if num_weeks < 1:
        return None # Not enough data for a trend

    first_week_df = df[df[date_col].between(first_week_start, first_week_end)]
    last_week_df = df[df[date_col].between(last_week_start, last_week_end)]

    results = []
    for metric in metrics:
        first_week_val = _calculate_metric(first_week_df, metric)
        last_week_val = _calculate_metric(last_week_df, metric)
        
        if first_week_val <= 0 or last_week_val <= 0:
            continue # CWGR is not meaningful for zero or negative start/end values

        # CWGR = ((End Value / Start Value)^(1 / N)) - 1
        cwgr = (last_week_val / first_week_val) ** (1 / num_weeks) - 1

        # Only include if the growth rate is significant
        if abs(cwgr * 100) > trend_config.get('significance_threshold_pct', 1.0):
            results.append({
                "metric_name": metric['name'],
                "cwgr": cwgr * 100, # As a percentage
                "description": f"Compound Weekly Growth Rate over {num_weeks:.0f} weeks was {cwgr:+.2%}"
            })

    return results 

def analyze_quarterly_trend(data: str, value_col: str, date_col: str, quarter_col: str = "quarter") -> str:
    """
    ADK tool wrapper for quarterly trend analysis.
    
    Args:
        data: CSV string containing the dataset
        value_col: Name of the value column
        date_col: Name of the date column
        quarter_col: Name of the quarter column (optional)
    
    Returns:
        String summary of the quarterly trend analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        if date_col not in df.columns:
            return f"Error: Date column '{date_col}' not found in data"
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract quarter if not provided
        if quarter_col not in df.columns:
            df['quarter'] = df[date_col].dt.quarter
            quarter_col = 'quarter'
        
        # Group by quarter
        quarterly_summary = df.groupby(quarter_col)[value_col].agg(['mean', 'sum', 'count'])
        
        result_summary = f"Quarterly Trend Analysis Results\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Date Column: {date_col}\n\n"
        
        result_summary += "Quarterly Summary:\n"
        for quarter, row in quarterly_summary.iterrows():
            result_summary += f"  Q{quarter}: Mean={row['mean']:.4f}, Sum={row['sum']:.4f}, Count={row['count']}\n"
        
        # Calculate quarter-over-quarter growth
        qoq_growth = quarterly_summary['mean'].pct_change()
        result_summary += f"\nQuarter-over-Quarter Growth:\n"
        for quarter, growth in qoq_growth.items():
            if not pd.isna(growth):
                result_summary += f"  Q{quarter}: {growth:.2%}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in quarterly trend analysis: {str(e)}" 