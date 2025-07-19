# data_analyzer/analysis/pop_change.py
import pandas as pd
from typing import Dict, Any, Union

def run_pop_analysis(df: pd.DataFrame, metric_col: str, period_col: str, current_period: Union[str, int], prior_period: Union[str, int]) -> Dict[str, Any]:
    """
    Analyze how a metric has changed between two periods.
    """
    current_data = df[df[period_col] == current_period]
    prior_data = df[df[period_col] == prior_period]

    current_value = current_data[metric_col].sum() if not current_data.empty else 0
    prior_value = prior_data[metric_col].sum() if not prior_data.empty else 0

    if prior_value != 0:
        absolute_change = current_value - prior_value
        percent_change = (absolute_change / prior_value) * 100
    else:
        absolute_change = current_value
        percent_change = float('inf')

    return {
        'current_period': current_period,
        'prior_period': prior_period,
        'current_value': current_value,
        'prior_value': prior_value,
        'absolute_change': absolute_change,
        'percent_change': f"{percent_change:.2f}%" if percent_change != float('inf') else "inf"
    }

def calculate_pop_change(data: str, current_col: str, prior_col: str, dimension_col: str = "") -> str:
    """
    ADK tool wrapper for calculating period-over-period change.
    
    Args:
        data: CSV string containing the dataset
        current_col: Name of the current period column
        prior_col: Name of the prior period column
        dimension_col: Name of the dimension column (optional)
    
    Returns:
        String summary of the period-over-period change analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if current_col not in df.columns:
            return f"Error: Current column '{current_col}' not found in data"
        if prior_col not in df.columns:
            return f"Error: Prior column '{prior_col}' not found in data"
        
        # Calculate change
        df['pop_change'] = df[current_col] - df[prior_col]
        df['pop_change_pct'] = ((df[current_col] - df[prior_col]) / df[prior_col]) * 100
        
        result_summary = f"Period-over-Period Change Analysis\n"
        result_summary += f"Current Column: {current_col}\n"
        result_summary += f"Prior Column: {prior_col}\n\n"
        
        # Overall change
        total_current = df[current_col].sum()
        total_prior = df[prior_col].sum()
        total_change = total_current - total_prior
        total_change_pct = ((total_current - total_prior) / total_prior) * 100 if total_prior != 0 else 0
        
        result_summary += f"Overall Change: {total_change:.4f} ({total_change_pct:.2f}%)\n"
        
        # By dimension if provided
        if dimension_col and dimension_col in df.columns:
            dim_change = df.groupby(dimension_col).agg({
                'pop_change': 'sum',
                'pop_change_pct': 'mean'
            }).sort_values('pop_change', ascending=False)
            
            result_summary += f"\nChange by {dimension_col}:\n"
            for idx, row in dim_change.head(10).iterrows():
                result_summary += f"  {idx}: {row['pop_change']:.4f} ({row['pop_change_pct']:.2f}%)\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in POP change analysis: {str(e)}"
