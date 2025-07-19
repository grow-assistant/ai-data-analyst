# data_analyzer/analysis/change_drivers.py
import pandas as pd
from typing import Dict, Any, Tuple, Optional

def _calculate_metric(df: pd.DataFrame, metric: Dict[str, str]) -> float:
    """Helper function to calculate a metric based on its aggregation method."""
    method = metric['aggregation_method']
    numerator_col = metric['numerator']
    
    if df.empty or numerator_col not in df.columns:
        return 0

    if method == 'sum':
        return df[numerator_col].sum()
    elif method == 'mean':
        return df[numerator_col].mean()
    elif method == 'ratio':
        denominator_col = metric.get('denominator')
        if not denominator_col or denominator_col not in df.columns:
            return 0
        
        num = df[numerator_col].sum()
        den = df[denominator_col].sum()
        return num / den if den != 0 else 0
    return 0

def _calculate_metric_by_group(df: pd.DataFrame, metric: Dict[str, str], dimension_col: str) -> pd.Series:
    """Helper function to calculate a metric, grouped by a dimension."""
    method = metric['aggregation_method']
    numerator_col = metric['numerator']
    
    if method == 'sum':
        return df.groupby(dimension_col)[numerator_col].sum()
    elif method == 'mean':
        return df.groupby(dimension_col)[numerator_col].mean()
    elif method == 'ratio':
        denominator_col = metric['denominator']
        grouped = df.groupby(dimension_col)[[numerator_col, denominator_col]].sum()
        return grouped[numerator_col] / grouped[denominator_col]
    return pd.Series()

def analyze_change_contributors(
    df: pd.DataFrame, 
    metric: Dict[str, str], 
    dimension_col: str, 
    date_col: str, 
    current_period: Tuple[pd.Timestamp, pd.Timestamp],
    prior_period: Tuple[pd.Timestamp, pd.Timestamp]
) -> Optional[Dict[str, Any]]:
    """
    Analyzes the contributors to a metric's change, returning the top positive
    and negative drivers.
    """
    if dimension_col not in df.columns:
        return None

    current_df = df[df[date_col].between(current_period[0], current_period[1])]
    prior_df = df[df[date_col].between(prior_period[0], prior_period[1])]

    current_agg = _calculate_metric_by_group(current_df, metric, dimension_col)
    prior_agg = _calculate_metric_by_group(prior_df, metric, dimension_col)
    
    change_df = pd.DataFrame({'current': current_agg, 'prior': prior_agg}).fillna(0)
    change_df['change'] = change_df['current'] - change_df['prior']
    
    if change_df.empty or change_df['change'].sum() == 0:
        return None

    # Sort by the absolute change to find the most impactful members
    change_df['abs_change'] = change_df['change'].abs()
    change_df = change_df.sort_values('abs_change', ascending=False)

    # Separate into positive and negative contributors
    positive_drivers = change_df[change_df['change'] > 0]
    negative_drivers = change_df[change_df['change'] < 0]

    # Return the top 3 of each, if they exist
    top_positive = positive_drivers.head(3).to_dict('index')
    top_negative = negative_drivers.head(3).to_dict('index')

    if not top_positive and not top_negative:
        return None

    return {
        "total_change": change_df['change'].sum(),
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative
    } 

def find_change_drivers(data: str, current_col: str, prior_col: str, dimension_cols: str = "", top_n: int = 5) -> str:
    """
    ADK tool wrapper for finding change drivers.
    
    Args:
        data: CSV string containing the dataset
        current_col: Name of the current period column
        prior_col: Name of the prior period column
        dimension_cols: Comma-separated list of dimension columns
        top_n: Number of top change drivers to return
    
    Returns:
        String summary of the change drivers analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if current_col not in df.columns:
            return f"Error: Current column '{current_col}' not found in data"
        if prior_col not in df.columns:
            return f"Error: Prior column '{prior_col}' not found in data"
        
        # Calculate change
        df['change'] = df[current_col] - df[prior_col]
        df['change_pct'] = ((df[current_col] - df[prior_col]) / df[prior_col]) * 100
        
        # Parse dimension columns
        dims = [col.strip() for col in dimension_cols.split(",") if col.strip()] if dimension_cols else []
        
        result_summary = f"Change Drivers Analysis\n"
        result_summary += f"Current Column: {current_col}\n"
        result_summary += f"Prior Column: {prior_col}\n"
        result_summary += f"Dimensions: {dims}\n\n"
        
        # Overall change
        total_current = df[current_col].sum()
        total_prior = df[prior_col].sum()
        total_change = total_current - total_prior
        total_change_pct = ((total_current - total_prior) / total_prior) * 100 if total_prior != 0 else 0
        
        result_summary += f"Overall Change: {total_change:.4f} ({total_change_pct:.2f}%)\n\n"
        
        # Change drivers by dimension
        if dims:
            for dim in dims:
                if dim in df.columns:
                    dim_change = df.groupby(dim)['change'].sum().sort_values(ascending=False)
                    
                    result_summary += f"=== {dim.upper()} CHANGE DRIVERS ===\n"
                    result_summary += f"Top {min(top_n, len(dim_change))} positive drivers:\n"
                    
                    for idx, change in dim_change.head(top_n).items():
                        result_summary += f"  {idx}: +{change:.4f}\n"
                    
                    result_summary += f"Top {min(top_n, len(dim_change))} negative drivers:\n"
                    for idx, change in dim_change.tail(top_n).items():
                        result_summary += f"  {idx}: {change:.4f}\n"
                    
                    result_summary += "\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in change drivers analysis: {str(e)}" 