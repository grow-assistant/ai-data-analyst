import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_top_drivers(impact_analysis: pd.DataFrame, direction: str, top_n: int = 5, impact_col: str = 'ratio_change_contrib') -> pd.DataFrame:
    """
    Returns the top dimension members that moved the metric in the same direction
    as the observed overall change.
    """
    if direction == "Positive":
        # Top drivers are those with positive contribution
        drivers = impact_analysis[impact_analysis[impact_col] > 0].copy()
        drivers = drivers.sort_values(impact_col, ascending=False).head(top_n)
    else:
        # Direction is Negative, top drivers are those with negative contribution
        drivers = impact_analysis[impact_analysis[impact_col] < 0].copy()
        drivers = drivers.sort_values(impact_col, ascending=True).head(top_n)

    # Calculate difference to next
    drivers = drivers.sort_values(impact_col, ascending=(direction == "Negative"))
    drivers['next_impact'] = drivers[impact_col].shift(-1)
    drivers['Diff_to_Next'] = drivers.apply(
        lambda row: row[impact_col] - row['next_impact'] if not pd.isna(row['next_impact']) else None,
        axis=1
    )

    return drivers

def print_dimension_drivers(df: pd.DataFrame) -> None:
    display_rows = []
    if df.empty:
        logger.info("No drivers found.")
        return
    dimension_col = df['dimension'].iloc[0] if 'dimension' in df.columns else 'Dimension'
    
    impact_col_name = 'Ratio Change Contrib' if 'ratio_change_contrib' in df.columns else 'Change'

    for _, row in df.iterrows():
        # Extract dimension name and value
        for col in df.columns:
            if col in ['dimension', 'next_impact', 'Diff_to_Next'] or col.endswith(('_prior', '_current', '_contrib')):
                continue
            # The first non-metric column in dimension_cols is the dimension
            category_value = row[col]
            
            row_data = {
                'Dimension': dimension_col,
                'Category': category_value,
            }

            if 'ratio_change_contrib' in row:
                row_data['Ratio Change Contrib'] = round(row['ratio_change_contrib'], 4)
            if 'change' in row:
                row_data['Change'] = round(row['change'], 2)
            if 'Diff_to_Next' in row:
                row_data['Diff to Next Impact'] = round(row['Diff_to_Next'], 4) if row['Diff_to_Next'] is not None else None
            
            display_rows.append(row_data)
            break
    display_df = pd.DataFrame(display_rows)
    logger.info(display_df.to_string(index=False))
    logger.info("")

def find_top_contributors(data: str, impact_col: str = "impact", direction: str = "Positive", top_n: int = 5) -> str:
    """
    ADK tool wrapper for finding top contributors analysis.
    
    Args:
        data: CSV string containing the impact analysis results
        impact_col: Name of the impact column
        direction: Direction to analyze ("Positive" or "Negative")
        top_n: Number of top contributors to return
    
    Returns:
        String summary of the top contributors
    """
    try:
        import io
        df = pd.read_csv(io.StringIO(data))
        
        if impact_col not in df.columns:
            return f"Error: Impact column '{impact_col}' not found in data"
        
        # Get top drivers
        drivers = get_top_drivers(df, direction, top_n, impact_col)
        
        result_summary = f"Top {top_n} Contributors Analysis\n"
        result_summary += f"Direction: {direction}\n"
        result_summary += f"Impact Column: {impact_col}\n\n"
        
        if not drivers.empty:
            result_summary += "Top Contributors:\n"
            for idx, row in drivers.iterrows():
                result_summary += f"- {idx}: {row[impact_col]:.4f}\n"
        else:
            result_summary += "No contributors found for the specified criteria.\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in top contributors analysis: {str(e)}"
