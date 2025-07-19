# data_analyzer/analysis/momentum.py
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from .timeframe import get_time_periods, get_reference_date_from_config
from .change_drivers import _calculate_metric

logger = logging.getLogger(__name__)

def calculate_momentum_analysis(df: pd.DataFrame, metrics: List[Dict[str, Any]], date_col: str, config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Calculates momentum for a list of metrics.
    Compares a 4-week rolling period to the prior 4-week period.
    """
    momentum_config = config.get('analysis', {}).get('momentum_analysis', {})
    if not momentum_config.get('enabled', False):
        return None

    logger.info("--- Calculating Momentum Analysis (4-week vs. prior 4-week) ---")
    
    # Define the two 4-week periods
    latest_date = get_reference_date_from_config(config, df, date_col)
    current_4_week_end = latest_date
    current_4_week_start = latest_date - pd.DateOffset(weeks=4)
    
    prior_4_week_end = current_4_week_start - pd.DateOffset(days=1)
    prior_4_week_start = prior_4_week_end - pd.DateOffset(weeks=4)
    
    current_period_df = df[df[date_col].between(current_4_week_start, current_4_week_end)]
    prior_period_df = df[df[date_col].between(prior_4_week_start, prior_4_week_end)]

    results = []
    for metric in metrics:
        current_val = _calculate_metric(current_period_df, metric)
        prior_val = _calculate_metric(prior_period_df, metric)

        change = current_val - prior_val
        pct_change = (change / prior_val) * 100 if prior_val != 0 else 0
        
        # Only include if the change is significant
        if abs(pct_change) > momentum_config.get('significance_threshold_pct', 5):
            results.append({
                "metric_name": metric['name'],
                "momentum_change_pct": pct_change,
                "description": f"4-week rolling average changed by {pct_change:.1f}% compared to the prior 4 weeks."
            })
            
    return results 

def calculate_momentum(data: str, value_col: str, date_col: str, periods: int = 3) -> str:
    """
    ADK tool wrapper for momentum calculation.
    
    Args:
        data: CSV string containing the dataset
        value_col: Name of the value column
        date_col: Name of the date column
        periods: Number of periods for momentum calculation
    
    Returns:
        String summary of the momentum analysis
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
        df = df.sort_values(date_col)
        
        # Calculate momentum (simple rate of change)
        df['momentum'] = df[value_col].pct_change(periods=periods)
        
        result_summary = f"Momentum Analysis Results\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Date Column: {date_col}\n"
        result_summary += f"Periods: {periods}\n\n"
        
        # Summary statistics
        momentum_mean = df['momentum'].mean()
        momentum_std = df['momentum'].std()
        
        result_summary += f"Average Momentum: {momentum_mean:.4f}\n"
        result_summary += f"Momentum Std Dev: {momentum_std:.4f}\n"
        result_summary += f"Momentum Trend: {'Accelerating' if momentum_mean > 0 else 'Decelerating'}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in momentum calculation: {str(e)}" 