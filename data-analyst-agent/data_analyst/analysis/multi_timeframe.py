# data_analyzer/analysis/multi_timeframe.py
import logging
import pandas as pd
from typing import Dict, Any, List, Optional

from .timeframe import get_time_periods, get_reference_date_from_config
from .change_drivers import _calculate_metric
from .impact_analysis import analyze_periods

logger = logging.getLogger(__name__)

def run_multi_timeframe_analysis(df: pd.DataFrame, config: Dict[str, Any], date_col: str, metrics: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Run analysis across multiple timeframes to provide comprehensive insights.
    """
    additional_config = config.get('analysis', {}).get('additional_timeframes', {})
    if not additional_config.get('enabled', False):
        return {}
    
    timeframes = additional_config.get('timeframes', [])
    if not timeframes:
        return {}
    
    logger.info("===== Running Multi-Timeframe Analysis =====")
    
    reference_date = get_reference_date_from_config(config, df, date_col)
    results = {}
    
    for timeframe_config in timeframes:
        timeframe_name = timeframe_config.get('name')
        timeframe = timeframe_config.get('timeframe')
        description = timeframe_config.get('description', '')
        
        if not timeframe_name or not timeframe:
            continue
            
        logger.info(f"--- Analyzing {timeframe_name} ({description}) ---")
        
        try:
            periods = get_time_periods(timeframe, reference_date)
            current_start, current_end = periods['current']
            prior_start, prior_end = periods['prior']
            
            # Filter data for the periods
            current_df = df[df[date_col].between(current_start, current_end)]
            prior_df = df[df[date_col].between(prior_start, prior_end)]
            
            if current_df.empty or prior_df.empty:
                logger.warning(f"Insufficient data for {timeframe_name} analysis")
                continue
            
            timeframe_results = {
                'description': description,
                'current_period': f"{current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}",
                'prior_period': f"{prior_start.strftime('%Y-%m-%d')} to {prior_end.strftime('%Y-%m-%d')}",
                'metrics': []
            }
            
            # Analyze each metric for this timeframe
            for metric in metrics:
                metric_name = metric['name']
                
                current_value = _calculate_metric(current_df, metric)
                prior_value = _calculate_metric(prior_df, metric)
                change = current_value - prior_value
                
                if prior_value != 0:
                    percent_change = (change / prior_value) * 100
                else:
                    percent_change = float('inf') if change > 0 else float('-inf') if change < 0 else 0
                
                # Get significance threshold
                threshold = metric.get('significance_threshold_pct', 0)
                is_significant = abs(percent_change) >= threshold if percent_change != float('inf') and percent_change != float('-inf') else True
                
                metric_result = {
                    'metric_name': metric_name,
                    'current_value': current_value,
                    'prior_value': prior_value,
                    'change': change,
                    'percent_change': percent_change,
                    'is_significant': is_significant,
                    'threshold': threshold
                }
                
                # If significant, get top contributors
                if is_significant and abs(change) > 1e-9:
                    # Run impact analysis to get top contributors
                    impact_results = analyze_periods(
                        df=df,
                        dimension_cols=['Region', 'Terminal'],  # Use key dimensions
                        numerator_col=metric.get('numerator'),
                        denominator_col=metric.get('denominator'),
                        date_col=date_col,
                        current_start=current_start,
                        current_end=current_end,
                        prior_start=prior_start,
                        prior_end=prior_end,
                        comparison_label=f"{timeframe_name} Analysis"
                    )
                    
                    if impact_results:
                        # Get top 3 positive and negative contributors
                        contributors = impact_results.get('contributors', [])
                        top_positive = [c for c in contributors if c.get('change', 0) > 0][:3]
                        top_negative = [c for c in contributors if c.get('change', 0) < 0][:3]
                        
                        metric_result['top_contributors'] = {
                            'positive': top_positive,
                            'negative': top_negative
                        }
                
                timeframe_results['metrics'].append(metric_result)
            
            results[timeframe_name] = timeframe_results
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe_name}: {e}")
            continue
    
    return results

def format_multi_timeframe_summary(results: Dict[str, Any]) -> str:
    """
    Format multi-timeframe analysis results into a readable summary.
    """
    if not results:
        return ""
    
    summary_lines = ["", "## Multi-Timeframe Analysis Summary", ""]
    
    for timeframe_name, timeframe_data in results.items():
        summary_lines.append(f"### {timeframe_name}")
        summary_lines.append(f"**Period:** {timeframe_data['description']}")
        summary_lines.append(f"**Current:** {timeframe_data['current_period']}")
        summary_lines.append(f"**Prior:** {timeframe_data['prior_period']}")
        summary_lines.append("")
        
        # Group metrics by significance
        significant_metrics = []
        flat_metrics = []
        
        for metric in timeframe_data['metrics']:
            if metric['is_significant']:
                significant_metrics.append(metric)
            else:
                flat_metrics.append(metric)
        
        if significant_metrics:
            summary_lines.append("**Significant Changes:**")
            for metric in significant_metrics:
                change_direction = "↑" if metric['change'] > 0 else "↓"
                pct_str = f"{metric['percent_change']:+.1f}%" if abs(metric['percent_change']) != float('inf') else "N/A"
                summary_lines.append(f"- {change_direction} **{metric['metric_name']}**: {pct_str}")
                
                # Add top contributors if available
                if 'top_contributors' in metric:
                    contributors = metric['top_contributors']
                    if contributors['positive']:
                        summary_lines.append(f"  - Top gains: {', '.join([c.get('dimension_value', 'Unknown') for c in contributors['positive']])}")
                    if contributors['negative']:
                        summary_lines.append(f"  - Top declines: {', '.join([c.get('dimension_value', 'Unknown') for c in contributors['negative']])}")
            summary_lines.append("")
        
        if flat_metrics:
            summary_lines.append("**Stable Metrics:**")
            for metric in flat_metrics:
                summary_lines.append(f"- **{metric['metric_name']}**: Minimal change ({metric['percent_change']:+.1f}%)")
            summary_lines.append("")
        
        summary_lines.append("---")
        summary_lines.append("")
    
    return "\n".join(summary_lines) 

def analyze_multi_timeframe(data: str, date_col: str, value_col: str, timeframes: str = "weekly,monthly") -> str:
    """
    ADK tool wrapper for multi-timeframe analysis.
    
    Args:
        data: CSV string containing the dataset
        date_col: Name of the date column
        value_col: Name of the value column
        timeframes: Comma-separated list of timeframes to analyze
    
    Returns:
        String summary of the multi-timeframe analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if date_col not in df.columns:
            return f"Error: Date column '{date_col}' not found in data"
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Parse timeframes
        tf_list = [tf.strip() for tf in timeframes.split(",")]
        
        result_summary = f"Multi-Timeframe Analysis Results\n"
        result_summary += f"Date Column: {date_col}\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Timeframes: {tf_list}\n\n"
        
        for timeframe in tf_list:
            result_summary += f"=== {timeframe.upper()} ANALYSIS ===\n"
            
            # Set up timeframe grouping
            if timeframe == "daily":
                df['period'] = df[date_col].dt.date
            elif timeframe == "weekly":
                df['period'] = df[date_col].dt.to_period('W')
            elif timeframe == "monthly":
                df['period'] = df[date_col].dt.to_period('M')
            elif timeframe == "quarterly":
                df['period'] = df[date_col].dt.to_period('Q')
            elif timeframe == "yearly":
                df['period'] = df[date_col].dt.to_period('Y')
            else:
                result_summary += f"Unsupported timeframe: {timeframe}\n\n"
                continue
            
            # Group by period
            period_summary = df.groupby('period')[value_col].agg(['mean', 'sum', 'count'])
            
            result_summary += f"Periods: {len(period_summary)}\n"
            result_summary += f"Average per period: {period_summary['mean'].mean():.4f}\n"
            result_summary += f"Total: {period_summary['sum'].sum():.4f}\n"
            result_summary += f"Volatility: {period_summary['mean'].std():.4f}\n\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in multi-timeframe analysis: {str(e)}" 