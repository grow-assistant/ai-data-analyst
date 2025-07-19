import pandas as pd
from typing import Optional, Any

def find_bottom_contributors(df: pd.DataFrame, metric_col: str, dimension_col: str, bottom_n: int = 5, period_col: Optional[str] = None, period_filter: Optional[Any] = None) -> pd.DataFrame:
    """
    Identify the bottom N contributors to a metric, optionally filtered by a period.
    """
    data = df.copy()

    # If a period filter is specified, apply it
    if period_col and period_filter is not None:
        data = data[data[period_col] == period_filter]

    # Aggregate metric values by dimension
    agg_df = data.groupby(dimension_col, as_index=False)[metric_col].sum()

    # Sort and return bottom N contributors
    agg_df = agg_df.sort_values(metric_col, ascending=True).head(bottom_n)
    return agg_df

def is_enabled() -> bool:
    """
    Indicates if the Bottom Contributors insight is currently enabled.
    """
    return True

def find_bottom_contributors(data: str, value_col: str, bottom_n: int = 5) -> str:
    """
    ADK tool wrapper for finding bottom contributors.
    
    Args:
        data: CSV string containing the dataset
        value_col: Name of the value column to analyze
        bottom_n: Number of bottom contributors to return
    
    Returns:
        String summary of the bottom contributors
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        
        # Get bottom contributors
        bottom_contributors = df.nsmallest(bottom_n, value_col)
        
        result_summary = f"Bottom {bottom_n} Contributors Analysis\n"
        result_summary += f"Value Column: {value_col}\n\n"
        
        if not bottom_contributors.empty:
            result_summary += "Bottom Contributors:\n"
            for idx, row in bottom_contributors.iterrows():
                result_summary += f"- Index {idx}: {row[value_col]:.4f}\n"
        else:
            result_summary += "No contributors found.\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in bottom contributors analysis: {str(e)}"
