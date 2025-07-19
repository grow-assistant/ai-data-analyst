# data_analyzer/analysis/narrative_drilldown.py
import logging
import pandas as pd
from typing import Dict, Any, List, Optional

from .timeframe import get_time_periods
from .change_drivers import analyze_change_contributors, _calculate_metric
from .outliers import is_member_an_outlier

logger = logging.getLogger(__name__)

def find_narrative_drilldown(
    df: pd.DataFrame,
    metric: Dict[str, Any],
    hierarchy: List[str],
    date_col: str,
    periods: Dict[str, tuple],
    outlier_config: Dict[str, Any],
    parent_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Recursively drills down through a dimensional hierarchy to find the root cause of a metric change.
    """
    narrative = []
    current_level = hierarchy[0]
    remaining_hierarchy = hierarchy[1:]
    
    current_period = periods['current']
    prior_period = periods['prior']

    # Analyze contributors at the current level
    contribution_analysis = analyze_change_contributors(
        df, metric, current_level, date_col, current_period, prior_period
    )

    if not contribution_analysis:
        return narrative

    # Determine if we're looking for the top gainer or loser based on the parent context
    # If a direction (positive/negative) is already established, stick with it.
    current_total_change = contribution_analysis.get('total_change', 0)
    
    # If this is the top-level call, the direction is determined by the total change.
    # Otherwise, inherit the direction from the parent.
    is_overall_positive = parent_context['is_positive_driver'] if parent_context else current_total_change > 0

    top_contributor_member = None
    top_contributor_data = None
    
    if is_overall_positive:
        contributors = contribution_analysis.get('top_positive_contributors', {})
        if contributors:
            top_contributor_member, top_contributor_data = list(contributors.items())[0]
    else: # Is overall negative
        detractors = contribution_analysis.get('top_negative_contributors', {})
        if detractors:
            top_contributor_member, top_contributor_data = list(detractors.items())[0]

    # If no driver is found in the expected direction, stop the drill-down for this path.
    if not top_contributor_member:
        return narrative

    # Check if the top contributor is an outlier
    metric_column_name = metric.get('numerator')
    if not metric_column_name:
        is_outlier = False # Cannot check outlier without a column
    else:
        is_outlier = is_member_an_outlier(
            df=df,
            dimension_col=current_level, # Corrected parameter name
            member_name=top_contributor_member, # Corrected parameter name
            metric_col=metric_column_name, # Pass the column name string
            threshold_factor=outlier_config.get('threshold_factor', 2.0)
        )

    narrative_step = {
        "level": current_level,
        "member": top_contributor_member,
        "change": top_contributor_data['change'],
        "contribution_pct": top_contributor_data.get('contribution_pct'),
        "is_outlier": is_outlier,
        "is_positive_driver": top_contributor_data['change'] > 0, # Base this on the actual change of the member
        "context": parent_context.get('text') if parent_context else None
    }
    narrative.append(narrative_step)

    # If there are more levels in the hierarchy, recurse
    if remaining_hierarchy and top_contributor_member:
        filtered_df = df[df[current_level] == top_contributor_member]
        new_context = {
            "text": f"within the {top_contributor_member} {current_level}",
            "is_positive_driver": is_overall_positive # Pass the established direction down
        }
        
        # Check if there is data to continue the drill-down
        if not filtered_df.empty:
            narrative.extend(
                find_narrative_drilldown(
                    df=filtered_df,
                    metric=metric,
                    hierarchy=remaining_hierarchy,
                    date_col=date_col,
                    periods=periods,
                    outlier_config=outlier_config,
                    parent_context=new_context
                )
            )

    return narrative 

def narrative_drilldown(data: str, metric_col: str, dimension_col: str, narrative_template: str = "default") -> str:
    """
    ADK tool wrapper for narrative drilldown analysis.
    
    Args:
        data: CSV string containing the dataset
        metric_col: Name of the metric column
        dimension_col: Name of the dimension column
        narrative_template: Template for narrative generation
    
    Returns:
        String narrative summary of the drilldown analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if metric_col not in df.columns:
            return f"Error: Metric column '{metric_col}' not found in data"
        if dimension_col not in df.columns:
            return f"Error: Dimension column '{dimension_col}' not found in data"
        
        # Group by dimension
        dim_summary = df.groupby(dimension_col)[metric_col].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        
        # Generate narrative
        total_value = dim_summary['sum'].sum()
        top_contributor = dim_summary.index[0]
        top_value = dim_summary.iloc[0]['sum']
        top_percentage = (top_value / total_value) * 100
        
        narrative = f"Narrative Drilldown Analysis\n"
        narrative += f"Metric: {metric_col}\n"
        narrative += f"Dimension: {dimension_col}\n\n"
        
        narrative += f"The analysis reveals that '{top_contributor}' is the largest contributor to {metric_col}, "
        narrative += f"accounting for {top_percentage:.1f}% of the total value ({top_value:.2f} out of {total_value:.2f}).\n\n"
        
        # Top 3 contributors
        narrative += "Top 3 contributors:\n"
        for i, (idx, row) in enumerate(dim_summary.head(3).iterrows()):
            percentage = (row['sum'] / total_value) * 100
            narrative += f"{i+1}. {idx}: {row['sum']:.2f} ({percentage:.1f}%)\n"
        
        return narrative
        
    except Exception as e:
        return f"Error in narrative drilldown: {str(e)}" 