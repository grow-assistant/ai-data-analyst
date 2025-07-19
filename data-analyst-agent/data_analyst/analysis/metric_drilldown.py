# data_analyzer/analysis/metric_drilldown.py
import logging
import pandas as pd
from typing import Dict, Any, List

from .timeframe import get_time_periods, get_reference_date_from_config
from .change_drivers import _calculate_metric
from .narrative_drilldown import find_narrative_drilldown
from .ratio_decomposition import analyze_ratio_decomposition

logger = logging.getLogger(__name__)

def run_metric_drilldown_analysis(df: pd.DataFrame, config: Dict[str, Any], date_col: str, metrics: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    For each metric, performs a narrative drill-down analysis to find the root cause of change.
    """
    analysis_results = []
    analysis_config = config.get('analysis', {})
    
    # Load config for time periods, drill-down, and outliers
    timeframe = analysis_config.get('trend_analysis', {}).get('timeframe', 'rolling_2_weeks')
    hierarchy = analysis_config.get('drilldown_hierarchy')
    if not hierarchy:
        logger.warning("`drilldown_hierarchy` not defined in config. Skipping narrative analysis.")
        return []
    
    reference_date = get_reference_date_from_config(config, df, date_col)
    periods = get_time_periods(timeframe, reference_date)
    outlier_config = analysis_config.get('outlier_analysis', {})

    logger.info("===== Starting Metric Drill-Down Analysis =====")

    for metric in metrics:
        metric_name = metric['name']
        logger.info(f"--- Analyzing Metric: {metric_name} ---")

        # Calculate overall trend for the metric
        current_df = df[df[date_col].between(periods['current'][0], periods['current'][1])]
        prior_df = df[df[date_col].between(periods['prior'][0], periods['prior'][1])]
        
        current_total = _calculate_metric(current_df, metric)
        prior_total = _calculate_metric(prior_df, metric)
        total_change = current_total - prior_total
        
        # --- Robustness Check for Floating Point Noise ---
        # If the absolute change is effectively zero, force it into the flat path immediately.
        is_effectively_flat = abs(total_change) < 1e-9

        # Check for significance
        threshold = metric.get('significance_threshold_pct', 0)
        percent_change = (total_change / prior_total) * 100 if prior_total != 0 else float('inf')

        result = {
            "metric": metric,
            "total_change": total_change,
            "current_value": current_total,
            "prior_value": prior_total,
            "percent_change": percent_change,
            "is_flat": False,
            "narrative": [],
            "decomposition": None
        }

        # For ratio metrics, perform a decomposition analysis
        if metric.get('aggregation_method') == 'ratio':
            result['decomposition'] = analyze_ratio_decomposition(
                current_df=current_df,
                prior_df=prior_df,
                metric=metric
            )

        # If change is not significant OR effectively flat, treat as flat and do a regional breakdown
        if is_effectively_flat or abs(percent_change) < threshold:
            # Re-check is_effectively_flat to ensure correct logging message
            if not is_effectively_flat:
                logger.info(f"Metric '{metric_name}' change ({percent_change:.2f}%) is below the {threshold}% significance threshold. Performing regional breakdown.")
            else:
                logger.info(f"Metric '{metric_name}' change is negligible. Performing regional breakdown.")

            result["is_flat"] = True
            
            regional_narratives = []
            top_level_dim = hierarchy[0]
            regions = df[top_level_dim].unique()

            for region in regions:
                region_df = df[df[top_level_dim] == region]
                if region_df.empty:
                    continue
                
                # We need to calculate the change just for this region to see if it's worth reporting
                region_current_df = current_df[current_df[top_level_dim] == region]
                region_prior_df = prior_df[prior_df[top_level_dim] == region]
                region_change = _calculate_metric(region_current_df, metric) - _calculate_metric(region_prior_df, metric)
                
                # Only generate a narrative if the regional change is meaningful
                if abs(region_change) > 1e-9:
                     regional_narratives.append({
                        "region": region,
                        "change": region_change,
                        "narrative": find_narrative_drilldown(
                            df=region_df,
                            metric=metric,
                            hierarchy=hierarchy[1:], # Start drilldown from the next level
                            date_col=date_col,
                            periods=periods,
                            outlier_config=outlier_config,
                            parent_context={
                                "text": f"for the {region} region",
                                "is_positive_driver": region_change > 0
                            }
                        )
                    })
            
            result["regional_narratives"] = regional_narratives
        else:
            # If the change is significant, run the full narrative drill-down from the top
            logger.info(f"Metric '{metric_name}' change is significant. Starting narrative drill-down.")
            result["narrative"] = find_narrative_drilldown(
                df=df,
                metric=metric,
                hierarchy=hierarchy,
                date_col=date_col,
                periods=periods,
                outlier_config=outlier_config
            )
        
        analysis_results.append(result)

    return analysis_results 

def drilldown_metric(data: str, metric_col: str, dimension_cols: str, top_n: int = 10) -> str:
    """
    ADK tool wrapper for metric drilldown analysis.
    
    Args:
        data: CSV string containing the dataset
        metric_col: Name of the metric column to drilldown
        dimension_cols: Comma-separated list of dimension columns
        top_n: Number of top items to show per dimension
    
    Returns:
        String summary of the metric drilldown analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if metric_col not in df.columns:
            return f"Error: Metric column '{metric_col}' not found in data"
        
        # Parse dimension columns
        dims = [col.strip() for col in dimension_cols.split(",") if col.strip()]
        
        result_summary = f"Metric Drilldown Analysis\n"
        result_summary += f"Metric Column: {metric_col}\n"
        result_summary += f"Dimensions: {dims}\n\n"
        
        for dim in dims:
            if dim not in df.columns:
                result_summary += f"Warning: Dimension '{dim}' not found in data\n"
                continue
            
            # Group by dimension and calculate metrics
            dim_summary = df.groupby(dim)[metric_col].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
            
            result_summary += f"=== {dim.upper()} DRILLDOWN ===\n"
            result_summary += f"Top {min(top_n, len(dim_summary))} by total:\n"
            
            for idx, row in dim_summary.head(top_n).iterrows():
                result_summary += f"  {idx}: Sum={row['sum']:.4f}, Mean={row['mean']:.4f}, Count={row['count']}\n"
            
            result_summary += f"Total for {dim}: {dim_summary['sum'].sum():.4f}\n\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in metric drilldown: {str(e)}" 