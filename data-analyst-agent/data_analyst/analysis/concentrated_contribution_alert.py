import pandas as pd
from typing import Dict, Any, Optional

def check_concentrated_contribution(df: pd.DataFrame, metric_col: str, dimension_col: str, threshold: float = 0.5, top_n_check: int = 3, period_col: Optional[str] = None, period_filter: Optional[Any] = None) -> Dict[str, Any]:
    """
    Checks if the top N dimension members make up for a certain threshold of the total.
    """
    data = df.copy()

    # If a period filter is specified, apply it
    if period_col and period_filter is not None:
        data = data[data[period_col] == period_filter]

    if data.empty:
        return {'is_concentrated': False, 'contribution_share': 0, 'top_contributors_df': pd.DataFrame()}

    # Aggregate metric values by dimension and sort
    agg_df = data.groupby(dimension_col, as_index=False)[metric_col].sum()
    agg_df = agg_df.sort_values(metric_col, ascending=False)

    total_contribution = agg_df[metric_col].sum()
    if total_contribution == 0:
        return {'is_concentrated': False, 'contribution_share': 0, 'top_contributors_df': pd.DataFrame()}

    # Get top N contributors and their contribution
    top_contributors_df = agg_df.head(top_n_check)
    top_contribution = top_contributors_df[metric_col].sum()
    contribution_share = top_contribution / total_contribution

    is_concentrated = contribution_share >= threshold

    return {
        'is_concentrated': is_concentrated,
        'contribution_share': contribution_share,
        'top_contributors_df': top_contributors_df,
        'checked_top_n': top_n_check
    }

def is_enabled() -> bool:
    """
    Indicates if the Concentrated Contribution Alert insight is currently enabled.
    """
    return True

def check_concentrated_contribution(data: str, value_col: str, dimension_col: str, threshold: float = 0.5) -> str:
    """
    ADK tool wrapper for concentrated contribution alert.
    
    Args:
        data: CSV string containing the dataset
        value_col: Name of the value column
        dimension_col: Name of the dimension column
        threshold: Concentration threshold (0-1) to trigger alert
    
    Returns:
        String summary of the concentration analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        if dimension_col not in df.columns:
            return f"Error: Dimension column '{dimension_col}' not found in data"
        
        # Group by dimension
        dim_summary = df.groupby(dimension_col)[value_col].sum().sort_values(ascending=False)
        total = dim_summary.sum()
        
        # Check concentration
        top_1 = dim_summary.iloc[0] / total if len(dim_summary) > 0 else 0
        top_3 = dim_summary.head(3).sum() / total if len(dim_summary) >= 3 else top_1
        top_5 = dim_summary.head(5).sum() / total if len(dim_summary) >= 5 else top_3
        
        result_summary = f"Concentration Alert Analysis\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Dimension: {dimension_col}\n"
        result_summary += f"Threshold: {threshold:.2%}\n\n"
        
        result_summary += f"Top 1 concentration: {top_1:.2%}\n"
        result_summary += f"Top 3 concentration: {top_3:.2%}\n"
        result_summary += f"Top 5 concentration: {top_5:.2%}\n\n"
        
        if top_1 > threshold:
            result_summary += f"ALERT: High concentration! Top item '{dim_summary.index[0]}' contributes {top_1:.2%}, exceeding threshold of {threshold:.2%}.\n"
        else:
            result_summary += "No high concentration alert.\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in concentration alert: {str(e)}"
