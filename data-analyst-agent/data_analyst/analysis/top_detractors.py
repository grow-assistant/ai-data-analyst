import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_top_detractors(impact_analysis: pd.DataFrame, direction: str, top_n: int = 5, impact_col: str = 'ratio_change_contrib') -> pd.DataFrame:
    """
    Returns the top dimension members that moved the metric in the opposite direction
    of the observed overall change.
    """
    if direction == "Positive":
        # Overall positive change, detractors have negative contribution
        detractors = impact_analysis[impact_analysis[impact_col] < 0].copy()
        detractors = detractors.sort_values(impact_col, ascending=True).head(top_n)
    else:
        # Overall negative change, detractors have positive contribution
        detractors = impact_analysis[impact_analysis[impact_col] > 0].copy()
        detractors = detractors.sort_values(impact_col, ascending=False).head(top_n)

    # Calculate difference to next
    detractors = detractors.sort_values(impact_col, ascending=(direction == "Negative"))
    detractors['next_impact'] = detractors[impact_col].shift(-1)
    detractors['Diff_to_Next'] = detractors.apply(
        lambda row: row[impact_col] - row['next_impact'] if not pd.isna(row['next_impact']) else None,
        axis=1
    )

    return detractors

def print_dimension_detractors(df: pd.DataFrame) -> None:
    display_rows = []
    if df.empty:
        logger.info("No detractors found.")
        return
    dimension_col = df['dimension'].iloc[0] if 'dimension' in df.columns else 'Dimension'

    for _, row in df.iterrows():
        for col in df.columns:
            if col in ['dimension', 'next_impact', 'Diff_to_Next'] or col.endswith(('_prior', '_current', '_contrib')):
                continue
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

def find_top_detractors(data: str, value_col: str, dimension_col: str, top_n: int = 5) -> str:
    """
    ADK tool wrapper for finding top detractors.
    
    Args:
        data: CSV string containing the dataset
        value_col: Name of the value column
        dimension_col: Name of the dimension column
        top_n: Number of top detractors to return
    
    Returns:
        String summary of the top detractors
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        if dimension_col not in df.columns:
            return f"Error: Dimension column '{dimension_col}' not found in data"
        
        # Group by dimension and find lowest performers
        dim_summary = df.groupby(dimension_col)[value_col].sum().sort_values(ascending=True)
        
        result_summary = f"Top {top_n} Detractors Analysis\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Dimension: {dimension_col}\n\n"
        
        if not dim_summary.empty:
            result_summary += "Top Detractors (lowest performers):\n"
            for idx, value in dim_summary.head(top_n).items():
                result_summary += f"- {idx}: {value:.4f}\n"
        else:
            result_summary += "No detractors found.\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in top detractors analysis: {str(e)}"
