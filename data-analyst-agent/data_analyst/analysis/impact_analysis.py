import pandas as pd
from data_analyst.analysis.find_top_contributors import get_top_drivers, print_dimension_drivers
from data_analyst.analysis.top_detractors import get_top_detractors, print_dimension_detractors
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def analyze_periods(df: pd.DataFrame, dimension_cols: List[str], numerator_col: str, denominator_col: Optional[str], date_col: str, 
                    current_start: pd.Timestamp, current_end: pd.Timestamp, prior_start: pd.Timestamp, prior_end: pd.Timestamp, comparison_label: str) -> Optional[Dict[str, Any]]:
    current_data = df[(df[date_col]>=current_start) & (df[date_col]<=current_end)]
    prior_data = df[(df[date_col]>=prior_start) & (df[date_col]<=prior_end)]

    if current_data.empty or prior_data.empty:
        logger.warning(f"No data available for {comparison_label}.")
        return None

    current_total_num = current_data[numerator_col].sum()
    prior_total_num = prior_data[numerator_col].sum()
    
    # Prepare results dictionary
    results = {
        "current_total": current_total_num,
        "prior_total": prior_total_num,
        "change": current_total_num - prior_total_num,
        "percent_change": ((current_total_num - prior_total_num) / prior_total_num * 100) if prior_total_num != 0 else 0
    }
    
    direction = "Positive" if results['change'] > 0 else "Negative"
    impact_col = 'change' # Default for standalone

    # Handle standalone metrics (no denominator)
    if denominator_col is None:
        logger.info(f"\n===== Overall Stats ({comparison_label}) =====")
        logger.info(f"Current Total: {int(current_total_num):,}")
        logger.info(f"Prior Total: {int(prior_total_num):,}")
        logger.info(f"Change: {int(current_total_num - prior_total_num):,}")
        logger.info(f"Percent Change: {((current_total_num - prior_total_num) / prior_total_num * 100):.1f}%\n")
        
        # For standalone metrics, skip ratio-based analysis
        logger.info(f"Skipping dimension analysis for standalone metric {numerator_col}\n")
        return results
    
    else:
        # Handle ratio metrics (with denominator)
        current_total_den = current_data[denominator_col].sum()
        current_ratio = current_total_num / current_total_den if current_total_den else 0

        prior_total_den = prior_data[denominator_col].sum()
        prior_ratio = prior_total_num / prior_total_den if prior_total_den else 0

        overall_ratio_diff = current_ratio - prior_ratio
        direction = "Positive" if overall_ratio_diff > 0 else "Negative"
        impact_col = 'ratio_change_contrib'

        # Add ratio information to results
        results.update({
            "current_ratio": current_ratio,
            "prior_ratio": prior_ratio,
            "ratio_difference": overall_ratio_diff
        })

        logger.info(f"\n===== Overall Stats ({comparison_label}) =====")
        logger.info(f"Current Numerator: {int(current_total_num):,}")
        logger.info(f"Current Denominator: {int(current_total_den):,}")
        logger.info(f"Current Ratio: {current_ratio:.2f}")
        logger.info(f"Prior Numerator: {int(prior_total_num):,}")
        logger.info(f"Prior Denominator: {int(prior_total_den):,}")
        logger.info(f"Prior Ratio: {prior_ratio:.2f}")
        logger.info(f"Overall Ratio Difference: {overall_ratio_diff:.2f}\n")

    all_dimension_drivers = []
    all_dimension_detractors = []

    for gb in dimension_cols:
        # Instead of using period-based logic, directly compute impacts from current_data and prior_data
        impact_analysis = calculate_dimension_impact(current_data, prior_data, [gb], numerator_col, denominator_col)
        if impact_analysis.empty:
            continue

        impact_analysis['dimension'] = gb

        drivers = get_top_drivers(impact_analysis, direction, impact_col=impact_col)
        if not drivers.empty:
            all_dimension_drivers.append(drivers)
            logger.info(f"--- Top Drivers: {gb} ({comparison_label}) ---")
            print_dimension_drivers(drivers)

        detractors = get_top_detractors(impact_analysis, direction, impact_col=impact_col)
        if not detractors.empty:
            all_dimension_detractors.append(detractors)
            logger.info(f"--- Top Detractors: {gb} ({comparison_label}) ---")
            print_dimension_detractors(detractors)

    results['drivers'] = pd.concat(all_dimension_drivers) if all_dimension_drivers else pd.DataFrame()
    results['detractors'] = pd.concat(all_dimension_detractors) if all_dimension_detractors else pd.DataFrame()

    return results

def calculate_dimension_impact(current_data: pd.DataFrame, prior_data: pd.DataFrame, dimension_cols: List[str], numerator_col: str, denominator_col: Optional[str]) -> pd.DataFrame:
    if current_data.empty or prior_data.empty:
        return pd.DataFrame()

    # Handle case where denominator_col is None
    if denominator_col is None:
        current_grp = current_data.groupby(dimension_cols, as_index=False).agg({numerator_col: 'sum'})
        prior_grp = prior_data.groupby(dimension_cols, as_index=False).agg({numerator_col: 'sum'})
        merged = pd.merge(current_grp, prior_grp, on=dimension_cols, how='outer', suffixes=('_current', '_prior')).fillna(0)
        merged['change'] = merged[f'{numerator_col}_current'] - merged[f'{numerator_col}_prior']
        merged = merged.sort_values('change', ascending=False)
        return merged

    current_grp = current_data.groupby(dimension_cols, as_index=False).agg({numerator_col: 'sum', denominator_col: 'sum'})
    prior_grp = prior_data.groupby(dimension_cols, as_index=False).agg({numerator_col: 'sum', denominator_col: 'sum'})

    current_grp['ratio'] = current_grp.apply(
        lambda row: row[numerator_col]/row[denominator_col] if row[denominator_col] != 0 else 0, axis=1
    )
    current_den_sum = current_grp[denominator_col].sum() or 1
    current_grp['den_share'] = current_grp[denominator_col] / current_den_sum

    prior_grp['ratio'] = prior_grp.apply(
        lambda row: row[numerator_col]/row[denominator_col] if row[denominator_col] != 0 else 0, axis=1
    )
    prior_den_sum = prior_grp[denominator_col].sum() or 1
    prior_grp['den_share'] = prior_grp[denominator_col] / prior_den_sum

    merged = pd.merge(prior_grp, current_grp, on=dimension_cols, how='outer', suffixes=('_prior', '_current')).fillna(0)
    merged['ratio_change_contrib'] = (
        (merged['ratio_current'] - merged['ratio_prior']) * merged['den_share_prior']
        + merged['ratio_current'] * (merged['den_share_current'] - merged['den_share_prior'])
    )

    dimension_current_total_den = merged[f"{denominator_col}_current"].sum() or 1
    merged['normalized_ratio_change_contrib'] = merged['ratio_change_contrib'] / dimension_current_total_den

    merged = merged.sort_values('ratio_change_contrib', ascending=False)
    return merged

def run_impact_analysis(df: pd.DataFrame, date_col: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run impact analysis based on configuration.
    
    Args:
        df: DataFrame with the data
        date_col: Name of the date column
        config: Configuration dictionary for impact analysis
    
    Returns:
        Dictionary with impact analysis results
    """
    try:
        results = {}
        
        # Get configuration parameters
        metrics = config.get('metrics', [])
        dimensions = config.get('dimensions', [])
        timeframes = config.get('timeframes', ['last_week'])
        
        if not metrics or not dimensions:
            logger.warning("Impact analysis requires metrics and dimensions to be configured.")
            return results
        
        # Import timeframe utilities
        from .timeframe import get_time_periods
        
        for timeframe in timeframes:
            periods = get_time_periods(timeframe, df[date_col].max())
            if not periods:
                continue
                
            timeframe_results = {}
            
            for metric in metrics:
                numerator_col = metric.get('numerator')
                denominator_col = metric.get('denominator')
                
                if not numerator_col or numerator_col not in df.columns:
                    continue
                
                if denominator_col and denominator_col not in df.columns:
                    denominator_col = None
                
                # Analyze current vs prior period
                analysis_result = analyze_periods(
                    df=df,
                    dimension_cols=dimensions,
                    numerator_col=numerator_col,
                    denominator_col=denominator_col,
                    date_col=date_col,
                    current_start=periods['current']['start'],
                    current_end=periods['current']['end'],
                    prior_start=periods['prior']['start'],
                    prior_end=periods['prior']['end'],
                    comparison_label=f"{timeframe}_{numerator_col}"
                )
                
                if analysis_result:
                    timeframe_results[numerator_col] = analysis_result
            
            if timeframe_results:
                results[timeframe] = timeframe_results
        
        return results
        
    except Exception as e:
        logger.error(f"Error in impact analysis: {str(e)}")
        return {}

def analyze_impact(data: str, numerator_col: str, denominator_col: str, dimension_cols: str = "", date_col: str = "date") -> str:
    """
    ADK tool wrapper for impact analysis.
    
    Args:
        data: CSV string containing the dataset
        numerator_col: Name of the numerator column
        denominator_col: Name of the denominator column
        dimension_cols: Comma-separated list of dimension columns
        date_col: Name of the date column
    
    Returns:
        String summary of the impact analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        # Parse dimension columns
        dims = [col.strip() for col in dimension_cols.split(",") if col.strip()] if dimension_cols else []
        
        # Check if required columns exist
        if numerator_col not in df.columns:
            return f"Error: Numerator column '{numerator_col}' not found in data"
        if denominator_col not in df.columns:
            return f"Error: Denominator column '{denominator_col}' not found in data"
        
        result_summary = f"Impact Analysis Results\n"
        result_summary += f"Numerator: {numerator_col}\n"
        result_summary += f"Denominator: {denominator_col}\n"
        result_summary += f"Dimensions: {dims}\n\n"
        
        # Basic impact calculation
        if len(dims) > 0:
            for dim in dims:
                if dim in df.columns:
                    # Group by dimension and calculate ratios
                    grouped = df.groupby(dim).agg({
                        numerator_col: 'sum',
                        denominator_col: 'sum'
                    })
                    grouped['ratio'] = grouped[numerator_col] / grouped[denominator_col]
                    
                    result_summary += f"Impact by {dim}:\n"
                    for idx, row in grouped.head().iterrows():
                        result_summary += f"  {idx}: {row['ratio']:.4f}\n"
                    result_summary += "\n"
        
        # Overall ratio
        total_ratio = df[numerator_col].sum() / df[denominator_col].sum() if df[denominator_col].sum() != 0 else 0
        result_summary += f"Overall Ratio: {total_ratio:.4f}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in impact analysis: {str(e)}"
