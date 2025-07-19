"""ADK tool wrappers for data analyst functions."""

from google.adk.tools import FunctionTool

# Import all the tool functions from analysis modules
from .analysis.trends import detect_trends
from .analysis.change_drivers import find_change_drivers
from .analysis.find_bottom_contributors import find_bottom_contributors
from .analysis.concentrated_contribution_alert import check_concentrated_contribution
from .analysis.metric_drilldown import drilldown_metric
from .analysis.multi_timeframe import analyze_multi_timeframe
from .analysis.momentum import calculate_momentum
from .analysis.metrics import calculate_metrics
from .analysis.impact_analysis import analyze_impact
from .analysis.narrative_drilldown import narrative_drilldown
from .analysis.find_top_contributors import find_top_contributors
from .analysis.outliers import detect_outliers
from .analysis.pop_change import calculate_pop_change
from .analysis.quarterly_trend import analyze_quarterly_trend
from .analysis.ratio_decomposition import decompose_ratio
from .analysis.top_detractors import find_top_detractors
from .analysis.timeframe import analyze_timeframe

# Create ADK FunctionTool instances for each analysis function
basic_stats_tool = FunctionTool(func=calculate_metrics)
correlation_tool = FunctionTool(func=detect_outliers)  # Using outliers as correlation proxy
ratio_decomposition_tool = FunctionTool(func=decompose_ratio)
impact_analysis_tool = FunctionTool(func=analyze_impact)
bottom_contributors_tool = FunctionTool(func=find_bottom_contributors)
top_contributors_tool = FunctionTool(func=find_top_contributors)
timeframe_tool = FunctionTool(func=analyze_timeframe)
multi_timeframe_tool = FunctionTool(func=analyze_multi_timeframe)
metric_drilldown_tool = FunctionTool(func=drilldown_metric)
quarterly_trend_tool = FunctionTool(func=analyze_quarterly_trend)
momentum_tool = FunctionTool(func=calculate_momentum)
trends_tool = FunctionTool(func=detect_trends)
metrics_tool = FunctionTool(func=calculate_metrics)
narrative_drilldown_tool = FunctionTool(func=narrative_drilldown)
change_drivers_tool = FunctionTool(func=find_change_drivers)
outliers_tool = FunctionTool(func=detect_outliers)
concentrated_alert_tool = FunctionTool(func=check_concentrated_contribution)
top_detractors_tool = FunctionTool(func=find_top_detractors)
pop_change_tool = FunctionTool(func=calculate_pop_change)

# List of all available tools for dynamic discovery
ALL_TOOLS = [
    basic_stats_tool,
    correlation_tool,
    ratio_decomposition_tool,
    impact_analysis_tool,
    bottom_contributors_tool,
    top_contributors_tool,
    timeframe_tool,
    multi_timeframe_tool,
    metric_drilldown_tool,
    quarterly_trend_tool,
    momentum_tool,
    trends_tool,
    metrics_tool,
    narrative_drilldown_tool,
    change_drivers_tool,
    outliers_tool,
    concentrated_alert_tool,
    top_detractors_tool,
    pop_change_tool,
] 