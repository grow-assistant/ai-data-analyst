# data_analyzer/analysis/ratio_decomposition.py
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def map_config_to_cache_columns(config_column_name: str) -> str:
    """
    Maps configuration column names to the abbreviated names used in cached data.
    """
    column_mapping = {
        "Line Haul Revenue Amount": "Total Revenue xFSR",
        "Truck Count": "Truck Count",
        "Loaded Traffic Miles": "Loaded Miles",
        "Total Miles": "Total Miles",
        "Idle Engine Time": "Idle Engine Time",
        "Driver Engine Time": "Driver Engine Time",
        "Order Count": "Revenue Order Count"
    }
    
    return column_mapping.get(config_column_name, config_column_name)

def detect_timeframe_columns(data: pd.DataFrame, numerator_col: str, denominator_col: str) -> Dict[str, List[str]]:
    """
    Dynamically detect available timeframe columns in the data.
    Returns a mapping of timeframe types to their column suffixes.
    """
    timeframes = {}
    
    # Common timeframe patterns
    patterns = {
        'wow': ['_Curr_Wk', '_Prior_Wk'],
        'yoy': ['_Curr_Yr', '_Prior_Yr'],
        'mom': ['_Curr_Mo', '_Prior_Mo'],
        'qoq': ['_Curr_Qtr', '_Prior_Qtr'],
        '4week': ['_Curr_4Wk', '_Prior_4Wk'],
        'rolling': ['_Current', '_Prior']
    }
    
    available_columns = set(data.columns)
    
    for timeframe_name, suffixes in patterns.items():
        required_cols = []
        for suffix in suffixes:
            required_cols.extend([
                f"{numerator_col}{suffix}",
                f"{denominator_col}{suffix}"
            ])
        
        # Check if all required columns exist
        if all(col in available_columns for col in required_cols):
            timeframes[timeframe_name] = suffixes
    
    return timeframes

def calculate_fleet_shares(data: pd.DataFrame, entity_col: str, truck_col: str) -> Dict[str, float]:
    """Calculate fleet share percentages for regions or terminals."""
    if entity_col not in data.columns or truck_col not in data.columns:
        return {}
    
    total_trucks = data[truck_col].sum()
    if total_trucks == 0:
        return {}
    
    fleet_shares = {}
    for _, row in data.iterrows():
        entity = row[entity_col]
        trucks = row[truck_col]
        fleet_shares[entity] = trucks / total_trucks
    
    return fleet_shares

def calculate_contribution_pp(entity_change_pct: float, fleet_share: float) -> float:
    """Calculate percentage point contribution to network change."""
    return entity_change_pct * fleet_share

def format_number(value: float, is_percentage: bool = False, is_currency: bool = False, is_large_number: bool = False) -> str:
    """Format numbers for executive presentation."""
    if pd.isna(value) or value == 0:
        return "0" if not is_percentage else "0%"
    
    if is_percentage:
        return f"{value:+.1f}%" if abs(value) >= 0.1 else f"{value:+.2f}%"
    elif is_currency:
        if abs(value) >= 1000000:
            return f"${value/1000000:.2f}M"
        elif abs(value) >= 1000:
            return f"${value/1000:.0f}K"
        else:
            return f"${value:.0f}"
    elif is_large_number:
        if abs(value) >= 1000000:
            return f"{value/1000000:.2f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.0f}K"
        else:
            return f"{value:.0f}"
    else:
        return f"{value:,.0f}"

def get_timeframe_display_names(timeframe1: str, timeframe2: str) -> Tuple[str, str]:
    """
    Generate human-readable display names for timeframe comparisons.
    """
    display_mapping = {
        'wow': 'Week-over-Week',
        'yoy': 'Year-over-Year', 
        'mom': 'Month-over-Month',
        'qoq': 'Quarter-over-Quarter',
        '4week': '4-Week Comparison',
        'rolling': 'Rolling Period'
    }
    
    return (
        display_mapping.get(timeframe1, timeframe1.upper()),
        display_mapping.get(timeframe2, timeframe2.upper())
    )

def extract_timeframe_data(data: pd.DataFrame, numerator_col: str, denominator_col: str, 
                          timeframe_suffixes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract current and prior period data for a given timeframe.
    """
    if len(timeframe_suffixes) != 2:
        raise ValueError("Timeframe must have exactly 2 suffixes (current and prior)")
    
    current_suffix, prior_suffix = timeframe_suffixes
    
    # Get total level data (first row)
    total_row = data.iloc[0] if not data.empty else None
    if total_row is None:
        return {}
    
    current_num = total_row.get(f"{numerator_col}{current_suffix}", 0)
    prior_num = total_row.get(f"{numerator_col}{prior_suffix}", 0)
    current_denom = total_row.get(f"{denominator_col}{current_suffix}", 0)
    prior_denom = total_row.get(f"{denominator_col}{prior_suffix}", 0)

    return {
        'current': {
            'numerator': current_num,
            'denominator': current_denom,
            'ratio': current_num / current_denom if current_denom != 0 else 0
        },
        'prior': {
            'numerator': prior_num,
            'denominator': prior_denom,
            'ratio': prior_num / prior_denom if prior_denom != 0 else 0
        }
    }

def create_dynamic_network_snapshot(data_dict: Dict[str, Dict[str, pd.DataFrame]], numerator_col: str, 
                                  denominator_col: str, timeframe1: str, timeframe2: str,
                                  metric_name: str, date: str) -> Dict[str, Any]:
    """
    Create a dynamic network snapshot comparing any two timeframes.
    """
    # Detect available timeframes
    available_timeframes = {}
    for tf_name, tf_data in data_dict.items():
        if isinstance(tf_data, dict) and 'total' in tf_data:
            df = tf_data['total']
            if not df.empty:
                tf_patterns = detect_timeframe_columns(df, numerator_col, denominator_col)
                available_timeframes.update(tf_patterns)
    
    # Validate requested timeframes
    if timeframe1 not in available_timeframes:
        return {"error": f"Timeframe '{timeframe1}' not available in data"}
    if timeframe2 not in available_timeframes:
        return {"error": f"Timeframe '{timeframe2}' not available in data"}
    
    # Extract data for both timeframes
    tf1_data = extract_timeframe_data(
        data_dict.get(f'{timeframe1}_analysis', {}).get('total', pd.DataFrame()),
        numerator_col, denominator_col, available_timeframes[timeframe1]
    )
    
    tf2_data = extract_timeframe_data(
        data_dict.get(f'{timeframe2}_analysis', {}).get('total', pd.DataFrame()),
        numerator_col, denominator_col, available_timeframes[timeframe2]
    )
    
    if not tf1_data or not tf2_data:
        return {"error": "Unable to extract timeframe data"}
    
    # Calculate changes for both timeframes
    def calculate_changes(data):
        current, prior = data['current'], data['prior']
        ratio_change_pct = ((current['ratio'] - prior['ratio']) / prior['ratio'] * 100) if prior['ratio'] != 0 else 0
        num_change_pct = ((current['numerator'] - prior['numerator']) / prior['numerator'] * 100) if prior['numerator'] != 0 else 0
        denom_change_pct = ((current['denominator'] - prior['denominator']) / prior['denominator'] * 100) if prior['denominator'] != 0 else 0
        
        return {
            'ratio_pct': ratio_change_pct,
            'numerator_pct': num_change_pct,
            'denominator_pct': denom_change_pct,
            'primary_driver': 'numerator' if abs(num_change_pct) > abs(denom_change_pct) else 'denominator'
        }
    
    tf1_changes = calculate_changes(tf1_data)
    tf2_changes = calculate_changes(tf2_data)
    
    # Get display names
    tf1_display, tf2_display = get_timeframe_display_names(timeframe1, timeframe2)

    return {
        "metric_name": metric_name,
        "date": date,
        "numerator_name": numerator_col.replace('_', ' ').title(),
        "denominator_name": denominator_col.replace('_', ' ').title(),
        "timeframe1": {
            "name": tf1_display,
            "key": timeframe1,
            "data": tf1_data,
            "changes": tf1_changes
        },
        "timeframe2": {
            "name": tf2_display,
            "key": timeframe2,
            "data": tf2_data,
            "changes": tf2_changes
        }
    }

def create_dynamic_regional_analysis(data_dict: Dict[str, Dict[str, pd.DataFrame]], numerator_col: str,
                                   denominator_col: str, timeframe1: str, timeframe2: str,
                                   network_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create regional contribution analysis for any two timeframes.
    """
    # Get regional data for both timeframes
    tf1_regional = data_dict.get(f'{timeframe1}_analysis', {}).get('by_region', pd.DataFrame())
    tf2_regional = data_dict.get(f'{timeframe2}_analysis', {}).get('by_region', pd.DataFrame())
    
    if tf1_regional.empty or tf2_regional.empty:
        return {"error": "No regional data available for specified timeframes"}
    
    # Detect timeframe patterns
    tf1_patterns = detect_timeframe_columns(tf1_regional, numerator_col, denominator_col)
    tf2_patterns = detect_timeframe_columns(tf2_regional, numerator_col, denominator_col)
    
    if timeframe1 not in tf1_patterns or timeframe2 not in tf2_patterns:
        return {"error": "Timeframe patterns not found in regional data"}
    
    def analyze_regional_timeframe(regional_data, timeframe_key, timeframe_patterns, network_change_pct):
        """Analyze regional contributions for a specific timeframe."""
        suffixes = timeframe_patterns[timeframe_key]
        current_suffix, prior_suffix = suffixes
        
        # Calculate fleet shares
        fleet_shares = calculate_fleet_shares(regional_data, 'Region', f"{denominator_col}{current_suffix}")
        
        regional_results = []
        for _, row in regional_data.iterrows():
            region = row.get('Region', 'Unknown')
            
            current_num = row.get(f"{numerator_col}{current_suffix}", 0)
            prior_num = row.get(f"{numerator_col}{prior_suffix}", 0)
            current_denom = row.get(f"{denominator_col}{current_suffix}", 0)
            prior_denom = row.get(f"{denominator_col}{prior_suffix}", 0)
            
            current_ratio = current_num / current_denom if current_denom != 0 else 0
            prior_ratio = prior_num / prior_denom if prior_denom != 0 else 0
            ratio_change_pct = ((current_ratio - prior_ratio) / prior_ratio * 100) if prior_ratio != 0 else 0
            
            fleet_share = fleet_shares.get(region, 0)
            contribution_pp = calculate_contribution_pp(ratio_change_pct / 100, fleet_share) * 100
            
            regional_results.append({
                "region": region,
                "fleet_share_pct": fleet_share * 100,
                "ratio_change_pct": ratio_change_pct,
                "contribution_pp": contribution_pp,
                "current_ratio": current_ratio,
                "prior_ratio": prior_ratio,
                "numerator_change": current_num - prior_num,
                "denominator_change": current_denom - prior_denom,
                "current_numerator": current_num,
                "current_denominator": current_denom
            })
        
        # Sort by contribution impact
        regional_results.sort(key=lambda x: abs(x["contribution_pp"]), reverse=True)
        return regional_results
    
    # Analyze both timeframes
    tf1_analysis = analyze_regional_timeframe(
        tf1_regional, timeframe1, tf1_patterns, 
        network_snapshot["timeframe1"]["changes"]["ratio_pct"]
    )
    
    tf2_analysis = analyze_regional_timeframe(
        tf2_regional, timeframe2, tf2_patterns,
        network_snapshot["timeframe2"]["changes"]["ratio_pct"]
    )
    
    return {
        "timeframe1_analysis": tf1_analysis,
        "timeframe2_analysis": tf2_analysis,
        "timeframe1_name": network_snapshot["timeframe1"]["name"],
        "timeframe2_name": network_snapshot["timeframe2"]["name"],
        "network_changes": {
            "timeframe1_pct": network_snapshot["timeframe1"]["changes"]["ratio_pct"],
            "timeframe2_pct": network_snapshot["timeframe2"]["changes"]["ratio_pct"]
        }
    }

def create_terminal_contribution_analysis(data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                        numerator_col: str, denominator_col: str,
                                        regional_analysis: Dict[str, Any],
                                        network_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Create Section 3: Terminal Contribution Analysis for the most impactful region."""

    # Find the region with the largest contribution in the first timeframe
    tf1_key = network_snapshot["timeframe1"]["key"]
    if not regional_analysis.get("timeframe1_analysis"):
        return {"error": "No regional analysis available for terminal drill-down"}
    
    primary_region_name = regional_analysis["timeframe1_analysis"][0]["region"]
    
    # Get terminal data for both timeframes
    tf1_terminal_data = data_dict.get(f'{network_snapshot["timeframe1"]["key"]}_analysis', {}).get('by_terminal', pd.DataFrame())
    tf2_terminal_data = data_dict.get(f'{network_snapshot["timeframe2"]["key"]}_analysis', {}).get('by_terminal', pd.DataFrame())

    if tf1_terminal_data.empty or tf2_terminal_data.empty:
        return {"error": f"No terminal data available for region '{primary_region_name}'"}
        
    # Filter for the primary region
    tf1_terminals = tf1_terminal_data[tf1_terminal_data['Region'] == primary_region_name].copy()
    tf2_terminals = tf2_terminal_data[tf2_terminal_data['Region'] == primary_region_name].copy()
    
    if tf1_terminals.empty or tf2_terminals.empty:
        return {"error": f"No terminal data for primary region '{primary_region_name}' found"}

    # Detect timeframe patterns
    tf1_patterns = detect_timeframe_columns(tf1_terminals, numerator_col, denominator_col)
    tf2_patterns = detect_timeframe_columns(tf2_terminals, numerator_col, denominator_col)

    if tf1_key not in tf1_patterns:
        return {"error": f"Timeframe patterns for {tf1_key} not found in terminal data"}

    def analyze_terminal_timeframe(terminal_data, timeframe_key, timeframe_patterns):
        current_suffix, prior_suffix = timeframe_patterns[timeframe_key]
        
        # Get total network denominator for contribution calculation
        network_denom = network_snapshot["timeframe1"]["data"]["current"]["denominator"]
        if network_denom == 0: return []
        
        terminal_results = []
        for _, row in terminal_data.iterrows():
            terminal = row.get('Terminal', 'Unknown')
            
            current_num = row.get(f"{numerator_col}{current_suffix}", 0)
            prior_num = row.get(f"{numerator_col}{prior_suffix}", 0)
            current_denom = row.get(f"{denominator_col}{current_suffix}", 0)
            prior_denom = row.get(f"{denominator_col}{prior_suffix}", 0)
            
            current_ratio = current_num / current_denom if current_denom != 0 else 0
            prior_ratio = prior_num / prior_denom if prior_denom != 0 else 0
            ratio_change_pct = ((current_ratio - prior_ratio) / prior_ratio * 100) if prior_ratio != 0 else 0
            
            # Contribution is based on the terminal's share of the total network denominator
            terminal_fleet_share = current_denom / network_denom
            contribution_pp = calculate_contribution_pp(ratio_change_pct / 100, terminal_fleet_share) * 100
            
            terminal_results.append({
                "terminal": terminal,
                "truck_count": current_denom,
                "numerator_change": current_num - prior_num,
                "ratio_change_pct": ratio_change_pct,
                "network_contribution_pp": contribution_pp,
            })
            
        terminal_results.sort(key=lambda x: abs(x["network_contribution_pp"]), reverse=True)
        return terminal_results

    tf1_analysis = analyze_terminal_timeframe(tf1_terminals, network_snapshot["timeframe1"]["key"], tf1_patterns)
    tf2_analysis = analyze_terminal_timeframe(tf2_terminals, network_snapshot["timeframe2"]["key"], tf2_patterns)

    return {
        "primary_region_name": primary_region_name,
        "timeframe1_analysis": tf1_analysis,
        "timeframe2_analysis": tf2_analysis,
    }

def generate_dynamic_executive_insights(network_snapshot: Dict[str, Any], 
                                      regional_analysis: Dict[str, Any],
                                      metric_name: str) -> Dict[str, Any]:
    """
    Generate executive insights for any metric and timeframe combination.
    """
    insights = {
        "network_health": [],
        "regional_drivers": [],
        "action_focus": [],
        "executive_summary": ""
    }
    
    # Extract key information
    tf1_name = network_snapshot["timeframe1"]["name"]
    tf2_name = network_snapshot["timeframe2"]["name"]
    tf1_change = network_snapshot["timeframe1"]["changes"]["ratio_pct"]
    tf2_change = network_snapshot["timeframe2"]["changes"]["ratio_pct"]
    tf1_driver = network_snapshot["timeframe1"]["changes"]["primary_driver"]
    tf2_driver = network_snapshot["timeframe2"]["changes"]["primary_driver"]
    
    numerator_name = network_snapshot["numerator_name"]
    denominator_name = network_snapshot["denominator_name"]
    
    # Network Health
    insights["network_health"].append(
        f"{tf1_name}: {metric_name} {'declined' if tf1_change < 0 else 'improved'} {abs(tf1_change):.1f}%, "
        f"primarily driven by {numerator_name.lower() if tf1_driver == 'numerator' else denominator_name.lower()} changes."
    )
    
    insights["network_health"].append(
        f"{tf2_name}: {metric_name} shows a {abs(tf2_change):.1f}% {'gain' if tf2_change > 0 else 'decline'}, "
        f"indicating {'positive' if tf2_change > 0 else 'concerning'} longer-term trends."
    )
    
    # Regional Drivers
    if regional_analysis.get("timeframe1_analysis"):
        top_tf1_region = regional_analysis["timeframe1_analysis"][0]
        insights["regional_drivers"].append(
            f"{tf1_name}: {top_tf1_region['region']} region is the primary driver "
            f"({top_tf1_region['contribution_pp']:+.1f} pp out of {tf1_change:+.1f} pp total)."
        )
        
        # Find offsetting regions for timeframe 1
        offsetting_regions = [r for r in regional_analysis["timeframe1_analysis"] 
                            if (r["contribution_pp"] > 0 and tf1_change < 0) or (r["contribution_pp"] < 0 and tf1_change > 0)]
        if offsetting_regions:
            offset_region = offsetting_regions[0]
            insights["regional_drivers"].append(
                f"{offset_region['region']} partially offset the impact ({offset_region['contribution_pp']:+.1f} pp)."
            )
    
    if regional_analysis.get("timeframe2_analysis"):
        top_tf2_region = regional_analysis["timeframe2_analysis"][0]
        insights["regional_drivers"].append(
            f"{tf2_name}: {top_tf2_region['region']} region leads the trend "
            f"({top_tf2_region['contribution_pp']:+.1f} pp contribution)."
        )
    
    # Action Focus
    if regional_analysis.get("timeframe1_analysis"):
        # Focus on the most impactful negative contributor for actionable insights
        negative_contributors = [r for r in regional_analysis["timeframe1_analysis"] if r["contribution_pp"] < 0]
        positive_contributors = [r for r in regional_analysis["timeframe1_analysis"] if r["contribution_pp"] > 0]
        
        if negative_contributors:
            region = negative_contributors[0]
            recovery_target = abs(region["numerator_change"]) * 0.7
            pp_impact = abs(region["contribution_pp"]) * 0.7
            insights["action_focus"].append(
                f"Recover {format_number(recovery_target, is_large_number=True)} {numerator_name.lower()} in {region['region']}: "
                f"would restore ~{pp_impact:.1f} pp of {metric_name.lower()} performance."
            )
        
        if positive_contributors:
            region = positive_contributors[0]
            insights["action_focus"].append(
                f"Replicate {region['region']}'s approach: "
                f"{format_number(region['numerator_change'], is_large_number=True)} {numerator_name.lower()} "
                f"contributed {region['contribution_pp']:+.1f} pp lift."
            )
    
    # Executive Summary
    summary_parts = []
    summary_parts.append(f"{metric_name} shows {abs(tf1_change):.1f}% {tf1_name.lower()} {'decline' if tf1_change < 0 else 'improvement'}")
    
    if regional_analysis.get("timeframe1_analysis"):
        primary_region = regional_analysis["timeframe1_analysis"][0]
        summary_parts.append(f"driven by {primary_region['region']} region")
    
    summary_parts.append(f"while {tf2_name.lower()} trend remains {'positive' if tf2_change > 0 else 'concerning'} at {tf2_change:+.1f}%")
    
    insights["executive_summary"] = ". ".join(summary_parts) + "."
    
    return insights

def format_analysis_to_markdown(analysis_results: Dict[str, Any]) -> str:
    """Formats the full decomposition analysis into a markdown report."""
    if analysis_results.get("error"):
        return f"### Analysis Error\n> {analysis_results['error']}"

    output = []
    
    # Header
    metric_name = analysis_results.get('metric_name', 'Unknown Metric')
    tf1_name = analysis_results['sections']['1_network_snapshot']['timeframe1']['name']
    tf2_name = analysis_results['sections']['1_network_snapshot']['timeframe2']['name']
    output.append(f"## {metric_name} – Decomposition & Contribution Analysis")
    output.append(f"**Time frames:** {tf1_name} vs. {tf2_name}")
    output.append(f"**Metrics used:** {analysis_results['numerator']}, {analysis_results['denominator']}, {metric_name}")
    output.append("")

    # Section 1: Network Snapshot
    output.append("### 1 | Network Snapshot")
    network = analysis_results['sections']['1_network_snapshot']
    numerator_name = network['numerator_name']
    denominator_name = network['denominator_name']
    tf1 = network['timeframe1']
    tf2 = network['timeframe2']

    header = f"| Metric | Current ({tf1['key']}) | Prior ({tf1['key']}) | Δ {tf1['key']} | Prior ({tf2['key']}) | Δ {tf2['key']} |"
    divider = "|:---|---:|---:|---:|---:|---:|"
    
    row1 = f"| **{metric_name}** | **{format_number(tf1['data']['current']['ratio'])}** | {format_number(tf1['data']['prior']['ratio'])} | **{format_number(tf1['changes']['ratio_pct'], is_percentage=True)}** | {format_number(tf2['data']['prior']['ratio'])} | **{format_number(tf2['changes']['ratio_pct'], is_percentage=True)}** |"
    row2 = f"| {numerator_name} | {format_number(tf1['data']['current']['numerator'], is_large_number=True)} | {format_number(tf1['data']['prior']['numerator'], is_large_number=True)} | {format_number(tf1['changes']['numerator_pct'], is_percentage=True)} | {format_number(tf2['data']['prior']['numerator'], is_large_number=True)} | {format_number(tf2['changes']['numerator_pct'], is_percentage=True)} |"
    row3 = f"| {denominator_name} | {format_number(tf1['data']['current']['denominator'])} | {format_number(tf1['data']['prior']['denominator'])} | {format_number(tf1['changes']['denominator_pct'], is_percentage=True)} | {format_number(tf2['data']['prior']['denominator'])} | {format_number(tf2['changes']['denominator_pct'], is_percentage=True)} |"
    
    output.extend([header, divider, row1, row2, row3, ""])

    # Section 2: Regional Contribution
    output.append("### 2 | Regional Contribution Analysis")
    regional = analysis_results['sections']['2_regional_contribution']
    if regional.get('error'):
        output.append(f"> {regional['error']}")
    else:
        # Timeframe 1 Table
        output.append(f"**{tf1['name']} ({format_number(regional['network_changes']['timeframe1_pct'], is_percentage=True)} total)**")
        reg_header = "| Region | Fleet Share | Δ M/T/W % | Contribution (pp) |"
        reg_divider = "|:---|---:|---:|---:|"
        output.extend([reg_header, reg_divider])
        total_pp = 0
        for r in regional['timeframe1_analysis']:
            total_pp += r['contribution_pp']
            output.append(f"| {r['region']} | {format_number(r['fleet_share_pct'], is_percentage=True)} | {format_number(r['ratio_change_pct'], is_percentage=True)} | {r['contribution_pp']:+.1f} pp |")
        output.append(f"| **Total** | **100%** | | **{total_pp:+.1f} pp** |")
        output.append("")

        # Timeframe 2 Table
        output.append(f"**{tf2['name']} ({format_number(regional['network_changes']['timeframe2_pct'], is_percentage=True)} total)**")
        output.extend([reg_header, reg_divider])
        total_pp = 0
        for r in regional['timeframe2_analysis']:
            total_pp += r['contribution_pp']
            output.append(f"| {r['region']} | {format_number(r['fleet_share_pct'], is_percentage=True)} | {format_number(r['ratio_change_pct'], is_percentage=True)} | {r['contribution_pp']:+.1f} pp |")
        output.append(f"| **Total** | **100%** | | **{total_pp:+.1f} pp** |")
        output.append("")

    # Section 3: Terminal Contribution
    terminal = analysis_results['sections'].get('3_terminal_contribution')
    if terminal and not terminal.get('error'):
        output.append(f"### 3 | Terminal Contribution Analysis – {terminal['primary_region_name']} Region")
        term_header = f"| {terminal['primary_region_name']} Terminal | Truck Cnt | Δ {numerator_name} | Δ M/T/W % | Netw. Contrib (pp) |"
        term_divider = "|:---|---:|---:|---:|---:|"
        output.extend([term_header, term_divider])
        for t in terminal['timeframe1_analysis'][:5]: # Top 5
             output.append(f"| {t['terminal']} | {t['truck_count']} | {format_number(t['numerator_change'], is_large_number=True)} | {format_number(t['ratio_change_pct'], is_percentage=True)} | {t['network_contribution_pp']:+.2f} pp |")
        output.append("")

    # Section 4: Insights
    insights = analysis_results['sections']['4_executive_insights']
    output.append("### 4 | CEO-Level Insights & Takeaways")
    output.append("**Network Health**")
    for insight in insights.get('network_health', []): output.append(f"- {insight}")
    output.append("\n**Regional Drivers**")
    for insight in insights.get('regional_drivers', []): output.append(f"- {insight}")
    output.append("\n**Action Focus**")
    for insight in insights.get('action_focus', []): output.append(f"- {insight}")
    output.append(f"\n**Summary:** {insights.get('executive_summary', '')}")
    
    return "\n".join(output)

def run_dynamic_ratio_decomposition_analysis(
    historical_data: Dict[str, Dict[str, pd.DataFrame]],
    metric_config: Dict[str, Any],
    timeframe1: str,
    timeframe2: str,
    date: str
) -> Dict[str, Any]:
    """
    Run dynamic ratio decomposition analysis for any metric and timeframe combination.
    
    Args:
        historical_data: Dictionary containing analysis data for different timeframes
        metric_config: Configuration for the metric (must have 'numerator' and 'denominator' keys)
        timeframe1: First timeframe for comparison (e.g., 'wow', 'mom', '4week')
        timeframe2: Second timeframe for comparison (e.g., 'yoy', 'qoq')
        date: Analysis date
    
    Returns:
        Dictionary containing the complete executive analysis
    """
    logger.info(f"Running dynamic ratio decomposition for {metric_config.get('name', 'Unknown Metric')} "
                f"comparing {timeframe1} vs {timeframe2} on {date}")
    
    # Validate inputs
    if not metric_config.get('numerator') or not metric_config.get('denominator'):
        return {"error": "Metric must have both 'numerator' and 'denominator' defined"}
    
    # Map column names
    numerator_col = map_config_to_cache_columns(metric_config['numerator'])
    denominator_col = map_config_to_cache_columns(metric_config['denominator'])
    metric_name = metric_config.get('name', f"{numerator_col}/{denominator_col}")
    
    try:
        # Create network snapshot
        network_snapshot = create_dynamic_network_snapshot(
            historical_data, numerator_col, denominator_col, 
            timeframe1, timeframe2, metric_name, date
        )
        
        if network_snapshot.get("error"):
            return network_snapshot
        
        # Create regional analysis
        regional_analysis = create_dynamic_regional_analysis(
            historical_data, numerator_col, denominator_col,
            timeframe1, timeframe2, network_snapshot
        )

        # Create terminal analysis
        terminal_analysis = create_terminal_contribution_analysis(
            historical_data, numerator_col, denominator_col,
            regional_analysis, network_snapshot
        )
        
        # Generate executive insights
        executive_insights = generate_dynamic_executive_insights(
            network_snapshot, regional_analysis, metric_name
        )
        
        return {
            "analysis_type": "Dynamic Executive Ratio Decomposition",
            "metric_name": metric_name,
            "numerator": numerator_col,
            "denominator": denominator_col,
            "timeframe1": timeframe1,
            "timeframe2": timeframe2,
            "date": date,
            "sections": {
                "1_network_snapshot": network_snapshot,
                "2_regional_contribution": regional_analysis,
                "3_terminal_contribution": terminal_analysis,
                "4_executive_insights": executive_insights
            },
            "markdown_report": format_analysis_to_markdown(
                {
                    "analysis_type": "Dynamic Executive Ratio Decomposition",
                    "metric_name": metric_name,
                    "numerator": numerator_col,
                    "denominator": denominator_col,
                    "timeframe1": timeframe1,
                    "timeframe2": timeframe2,
                    "date": date,
                    "sections": {
                        "1_network_snapshot": network_snapshot,
                        "2_regional_contribution": regional_analysis,
                        "3_terminal_contribution": terminal_analysis,
                        "4_executive_insights": executive_insights
                    },
                    "markdown_report": ""
                }
            )
        }
        
    except Exception as e:
        logger.error(f"Error in dynamic ratio decomposition analysis: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

# Backward compatibility - updated to use dynamic analysis
def run_executive_ratio_decomposition_analysis(
    historical_data: Dict[str, Dict[str, pd.DataFrame]],
    metric: Dict[str, Any],
    date: str
) -> Dict[str, Any]:
    """Backward compatibility wrapper - defaults to WoW vs YoY comparison."""
    return run_dynamic_ratio_decomposition_analysis(
        historical_data, metric, 'wow', 'yoy', date
    )

def run_ratio_decomposition_analysis(
    historical_data: Dict[str, Dict[str, pd.DataFrame]],
    metric: Dict[str, Any],
    date: str
) -> Dict[str, Any]:
    """Backward compatibility wrapper."""
    result = run_dynamic_ratio_decomposition_analysis(
        historical_data, metric, 'wow', 'yoy', date
    )
    if not result.get("error"):
        result["markdown_report"] = format_analysis_to_markdown(result)
    return result

# Legacy functions for backward compatibility
def analyze_ratio_decomposition_for_level(data: pd.DataFrame, metric: Dict[str, Any], 
                                        level_name: str, analysis_type: str, date: str) -> Dict[str, Any]:
    """Legacy function - use run_dynamic_ratio_decomposition_analysis instead."""
    return {"legacy": "Use run_dynamic_ratio_decomposition_analysis for full dynamic analysis"}

def analyze_ratio_decomposition(wow_data: pd.DataFrame, metric: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function - use run_dynamic_ratio_decomposition_analysis instead."""
    return {"legacy": "Use run_dynamic_ratio_decomposition_analysis for full dynamic analysis"} 

def decompose_ratio(data: str, numerator_col: str, denominator_col: str, dimension_cols: str = "", timeframe: str = "wow") -> str:
    """
    ADK tool wrapper for ratio decomposition analysis.
    
    Args:
        data: CSV string containing the dataset
        numerator_col: Name of the numerator column
        denominator_col: Name of the denominator column  
        dimension_cols: Comma-separated list of dimension columns
        timeframe: Timeframe for analysis (wow, yoy, mom, etc.)
    
    Returns:
        String summary of the ratio decomposition results
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
        
        # Detect available timeframe columns
        timeframe_cols = detect_timeframe_columns(df, numerator_col, denominator_col)
        
        if timeframe not in timeframe_cols:
            available = ", ".join(timeframe_cols.keys())
            return f"Error: Timeframe '{timeframe}' not available. Available: {available}"
        
        # Perform ratio decomposition (simplified version)
        result_summary = f"Ratio Decomposition Analysis for {numerator_col}/{denominator_col}\n"
        result_summary += f"Timeframe: {timeframe}\n"
        result_summary += f"Dimensions analyzed: {dims}\n"
        
        # Basic ratio calculation for current and prior periods
        current_num_col = f"{numerator_col}_Curr_{timeframe.upper()}"
        current_den_col = f"{denominator_col}_Curr_{timeframe.upper()}"
        prior_num_col = f"{numerator_col}_Prior_{timeframe.upper()}"
        prior_den_col = f"{denominator_col}_Prior_{timeframe.upper()}"
        
        if all(col in df.columns for col in [current_num_col, current_den_col, prior_num_col, prior_den_col]):
            current_ratio = df[current_num_col].sum() / df[current_den_col].sum() if df[current_den_col].sum() != 0 else 0
            prior_ratio = df[prior_num_col].sum() / df[prior_den_col].sum() if df[prior_den_col].sum() != 0 else 0
            ratio_change = current_ratio - prior_ratio
            
            result_summary += f"Current Period Ratio: {current_ratio:.4f}\n"
            result_summary += f"Prior Period Ratio: {prior_ratio:.4f}\n"
            result_summary += f"Ratio Change: {ratio_change:.4f}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in ratio decomposition: {str(e)}" 