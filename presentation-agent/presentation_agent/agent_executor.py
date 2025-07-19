"""
Enhanced Presentation Agent Executor with Google Gemini Executive Reporting
This module creates professional business reports using Google's Generative AI.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
from datetime import datetime
import json
from jinja2 import Template

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager
from common_utils.config import Settings

# Google Generative AI imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Google Gemini AI available - executive reporting enabled")
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Gemini AI not available - falling back to basic reporting")

logger = logging.getLogger(__name__)

class EnhancedPresentationExecutor:
    """
    Enhanced Presentation Agent with Google Gemini executive reporting capabilities.
    Creates professional business intelligence reports and insights.
    """
    
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        
        # Initialize Gemini if available
        if GEMINI_AVAILABLE:
            try:
                settings = Settings()
                genai.configure(api_key=settings.google_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                logger.info("✅ Google Gemini configured for executive reporting")
            except Exception as e:
                logger.warning(f"⚠️ Could not configure Gemini: {e}")
                self.model = None
        else:
            self.model = None
        
        logger.info("Enhanced Presentation Agent initialized with executive reporting capabilities")

    async def create_executive_report_skill(self, 
                                          analysis_handle_id: str, 
                                          report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a comprehensive executive report using Google Gemini for insights.
        """
        logger.info(f"Creating executive report for analysis handle: {analysis_handle_id}")
        
        try:
            config = report_config or {}
            
            # Get analysis results
            analysis_data = self.data_manager.get_data(analysis_handle_id)
            if not analysis_data:
                raise ValueError(f"Analysis data not found: {analysis_handle_id}")
            
            # Generate executive summary using Gemini
            executive_summary = await self._generate_executive_summary(analysis_data, config)
            
            # Create visualizations
            visualizations = self._create_comprehensive_visualizations(analysis_data)
            
            # Generate business insights using Gemini
            business_insights = await self._generate_business_insights(analysis_data, config)
            
            # Generate recommendations using Gemini
            strategic_recommendations = await self._generate_strategic_recommendations(analysis_data, config)
            
            # Create executive report HTML
            report_html = self._create_executive_report_html(
                executive_summary=executive_summary,
                business_insights=business_insights,
                strategic_recommendations=strategic_recommendations,
                visualizations=visualizations,
                analysis_data=analysis_data,
                config=config
            )
            
            # Create report handle
            report_handle = self.data_manager.create_handle(
                data=report_html,
                data_type="executive_report",
                metadata={
                    "original_analysis_handle": analysis_handle_id,
                    "report_type": "executive",
                    "generated_with_gemini": self.model is not None,
                    "generated_at": datetime.now().isoformat(),
                    "sections": ["executive_summary", "business_insights", "visualizations", "recommendations"]
                }
            )
            
            logger.info(f"Executive report generated successfully: {report_handle.handle_id}")
            
            return {
                "status": "completed",
                "report_data_handle_id": report_handle.handle_id,
                "report_type": "executive",
                "ai_powered": self.model is not None,
                "summary": {
                    "sections_generated": 4,
                    "visualizations_created": len(visualizations),
                    "insights_count": len(business_insights.get("key_insights", [])),
                    "recommendations_count": len(strategic_recommendations.get("recommendations", []))
                }
            }
            
        except Exception as e:
            logger.exception(f"Error creating executive report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_handle_id": analysis_handle_id
            }

    async def create_report_skill(self, data_handle_id: str, report_type: str = "executive") -> Dict[str, Any]:
        """
        Backward compatible report creation skill.
        """
        if report_type == "executive":
            return await self.create_executive_report_skill(data_handle_id)
        else:
            # Fall back to basic report
            return await self._create_basic_report(data_handle_id, report_type)

    async def _generate_executive_summary(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary using Google Gemini."""
        
        if not self.model:
            return self._generate_fallback_executive_summary(analysis_data)
        
        try:
            # Prepare analysis data for Gemini
            summary_prompt = self._create_executive_summary_prompt(analysis_data, config)
            
            response = await self.model.generate_content_async(summary_prompt)
            
            # Parse Gemini response
            summary_text = response.text
            
            return {
                "summary_text": summary_text,
                "key_findings": self._extract_key_findings(analysis_data),
                "performance_indicators": self._extract_performance_indicators(analysis_data),
                "generated_with_ai": True
            }
            
        except Exception as e:
            logger.warning(f"Gemini executive summary failed, using fallback: {e}")
            return self._generate_fallback_executive_summary(analysis_data)

    async def _generate_business_insights(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate business insights using Google Gemini."""
        
        if not self.model:
            return self._generate_fallback_insights(analysis_data)
        
        try:
            insights_prompt = self._create_business_insights_prompt(analysis_data, config)
            
            response = await self.model.generate_content_async(insights_prompt)
            insights_text = response.text
            
            return {
                "insights_narrative": insights_text,
                "key_insights": self._extract_structured_insights(insights_text),
                "market_observations": self._extract_market_observations(analysis_data),
                "generated_with_ai": True
            }
            
        except Exception as e:
            logger.warning(f"Gemini insights generation failed, using fallback: {e}")
            return self._generate_fallback_insights(analysis_data)

    async def _generate_strategic_recommendations(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendations using Google Gemini."""
        
        if not self.model:
            return self._generate_fallback_recommendations(analysis_data)
        
        try:
            recommendations_prompt = self._create_recommendations_prompt(analysis_data, config)
            
            response = await self.model.generate_content_async(recommendations_prompt)
            recommendations_text = response.text
            
            return {
                "recommendations_narrative": recommendations_text,
                "recommendations": self._extract_structured_recommendations(recommendations_text),
                "priority_actions": self._extract_priority_actions(analysis_data),
                "generated_with_ai": True
            }
            
        except Exception as e:
            logger.warning(f"Gemini recommendations generation failed, using fallback: {e}")
            return self._generate_fallback_recommendations(analysis_data)

    def _create_executive_summary_prompt(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create prompt for executive summary generation."""
        
        metadata = analysis_data.get("metadata", {})
        basic_stats = analysis_data.get("basic_statistics", {})
        key_metrics = analysis_data.get("key_metrics", {})
        trends = analysis_data.get("trend_analysis", {})
        
        prompt = f"""
As a senior business analyst, create a compelling executive summary based on the following data analysis:

DATASET OVERVIEW:
- Records analyzed: {metadata.get('data_shape', ['N/A', 'N/A'])[0]}
- Variables examined: {metadata.get('data_shape', ['N/A', 'N/A'])[1]}
- Analysis timestamp: {metadata.get('analysis_timestamp', 'N/A')}

KEY METRICS:
{json.dumps(key_metrics, indent=2) if key_metrics else "No specific metrics available"}

TREND ANALYSIS:
{json.dumps(trends, indent=2) if trends else "No trend analysis available"}

BASIC STATISTICS:
{json.dumps(basic_stats, indent=2) if basic_stats else "No basic statistics available"}

Please write a professional 2-3 paragraph executive summary that:
1. Highlights the most critical business insights
2. Quantifies key performance indicators where possible
3. Identifies significant trends or patterns
4. Uses business language appropriate for C-level executives
5. Focuses on actionable intelligence rather than technical details

Write in a confident, data-driven tone that demonstrates clear understanding of business implications.
"""
        return prompt

    def _create_business_insights_prompt(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create prompt for business insights generation."""
        
        impact_analysis = analysis_data.get("impact_analysis", {})
        contribution_analysis = analysis_data.get("contribution_analysis", {})
        outliers = analysis_data.get("outlier_detection", {})
        narrative_insights = analysis_data.get("narrative_insights", [])
        
        prompt = f"""
As an expert business intelligence analyst, provide deep business insights based on this comprehensive data analysis:

IMPACT ANALYSIS:
{json.dumps(impact_analysis, indent=2) if impact_analysis else "No impact analysis available"}

CONTRIBUTION ANALYSIS:
{json.dumps(contribution_analysis, indent=2) if contribution_analysis else "No contribution analysis available"}

OUTLIER DETECTION:
{json.dumps(outliers, indent=2) if outliers else "No outlier analysis available"}

NARRATIVE INSIGHTS:
{json.dumps(narrative_insights, indent=2) if narrative_insights else "No narrative insights available"}

Please provide strategic business insights that:
1. Identify the key drivers of performance
2. Explain what the outliers and anomalies mean for the business
3. Highlight hidden patterns that could impact strategy
4. Connect data findings to potential market opportunities or risks
5. Translate statistical findings into business implications

Format your response as clear, actionable insights that would be valuable for strategic decision-making.
"""
        return prompt

    def _create_recommendations_prompt(self, analysis_data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create prompt for strategic recommendations."""
        
        business_recommendations = analysis_data.get("business_recommendations", [])
        momentum_analysis = analysis_data.get("momentum_analysis", {})
        contribution_analysis = analysis_data.get("contribution_analysis", {})
        
        prompt = f"""
As a strategic business consultant, provide actionable recommendations based on this data analysis:

EXISTING RECOMMENDATIONS:
{json.dumps(business_recommendations, indent=2) if business_recommendations else "No existing recommendations"}

MOMENTUM ANALYSIS:
{json.dumps(momentum_analysis, indent=2) if momentum_analysis else "No momentum analysis available"}

CONTRIBUTION ANALYSIS:
{json.dumps(contribution_analysis, indent=2) if contribution_analysis else "No contribution analysis available"}

Please provide strategic recommendations that:
1. Are specific and actionable
2. Prioritize recommendations by potential impact and feasibility
3. Include short-term (next 30 days) and long-term (next quarter) actions
4. Address both opportunities and risks identified in the analysis
5. Consider resource requirements and implementation complexity

Format as:
- HIGH PRIORITY: Most critical actions for immediate implementation
- MEDIUM PRIORITY: Important strategic moves for the next quarter
- LONG-TERM: Strategic initiatives for sustained growth

Each recommendation should include the business rationale and expected outcomes.
"""
        return prompt

    def _create_comprehensive_visualizations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive visualizations for the executive report."""
        visualizations = []
        
        try:
            # Basic statistics visualization
            basic_stats = analysis_data.get("basic_statistics", {})
            if basic_stats.get("numeric_summary"):
                vis = self._create_metrics_overview_chart(basic_stats["numeric_summary"])
                visualizations.append(vis)
            
            # Key metrics visualization
            key_metrics = analysis_data.get("key_metrics", {})
            if key_metrics.get("business_metrics"):
                vis = self._create_key_metrics_dashboard(key_metrics["business_metrics"])
                visualizations.append(vis)
            
            # Contribution analysis visualization
            contribution = analysis_data.get("contribution_analysis", {})
            if contribution.get("dimension_contributions"):
                vis = self._create_contribution_analysis_chart(contribution["dimension_contributions"])
                visualizations.append(vis)
            
            # Trend analysis visualization
            trends = analysis_data.get("trend_analysis", {})
            if trends:
                vis = self._create_trend_summary_chart(trends)
                visualizations.append(vis)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations

    def _create_metrics_overview_chart(self, numeric_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create metrics overview chart."""
        try:
            # Extract means for overview
            means = {col: stats.get("mean", 0) for col, stats in numeric_summary.items() if isinstance(stats, dict)}
            
            if not means:
                return {"error": "No numeric data for overview chart"}
            
            fig = px.bar(
                x=list(means.keys()),
                y=list(means.values()),
                title="Key Metrics Overview",
                labels={"x": "Metrics", "y": "Average Values"}
            )
            
            fig.update_layout(
                title_font_size=16,
                showlegend=False,
                height=400
            )
            
            return {
                "title": "Key Metrics Overview",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "bar_chart"
            }
            
        except Exception as e:
            return {"error": f"Failed to create metrics overview: {e}"}

    def _create_key_metrics_dashboard(self, business_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create key metrics dashboard."""
        try:
            # Create subplot with metrics
            metrics_names = list(business_metrics.keys())
            totals = [metrics.get("total", 0) for metrics in business_metrics.values()]
            averages = [metrics.get("average", 0) for metrics in business_metrics.values()]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Total Values", "Average Values"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=metrics_names, y=totals, name="Total"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=metrics_names, y=averages, name="Average"),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="Business Metrics Dashboard",
                showlegend=False,
                height=400
            )
            
            return {
                "title": "Business Metrics Dashboard",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "dashboard"
            }
            
        except Exception as e:
            return {"error": f"Failed to create metrics dashboard: {e}"}

    def _create_contribution_analysis_chart(self, dimension_contributions: Dict[str, Any]) -> Dict[str, Any]:
        """Create contribution analysis chart."""
        try:
            # Use the first dimension for visualization
            first_dim = next(iter(dimension_contributions.keys()))
            top_contributors = dimension_contributions[first_dim].get("top_contributors", {})
            
            if not top_contributors:
                return {"error": "No contribution data available"}
            
            fig = px.pie(
                values=list(top_contributors.values()),
                names=list(top_contributors.keys()),
                title=f"Top Contributors - {first_dim}"
            )
            
            fig.update_layout(height=400)
            
            return {
                "title": f"Contribution Analysis - {first_dim}",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "pie_chart"
            }
            
        except Exception as e:
            return {"error": f"Failed to create contribution chart: {e}"}

    def _create_trend_summary_chart(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Create trend summary visualization."""
        try:
            # Create a simple trend indicator
            trend_direction = trends.get("trend_direction", "unknown")
            trend_strength = trends.get("trend_strength", 0)
            
            # Create gauge chart for trend strength
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = trend_strength * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Trend Strength ({trend_direction})"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if trend_direction == "increasing" else "darkred"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            return {
                "title": "Trend Analysis Summary",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "gauge"
            }
            
        except Exception as e:
            return {"error": f"Failed to create trend chart: {e}"}

    def _create_executive_report_html(self, 
                                    executive_summary: Dict[str, Any],
                                    business_insights: Dict[str, Any],
                                    strategic_recommendations: Dict[str, Any],
                                    visualizations: List[Dict[str, Any]],
                                    analysis_data: Dict[str, Any],
                                    config: Dict[str, Any]) -> str:
        """Create comprehensive executive report HTML."""
        
        # HTML template for executive report
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Business Intelligence Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border-left: 4px solid #007acc;
            background-color: #f8f9fa;
        }
        .section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 15px;
        }
        .section h3 {
            color: #34495e;
            font-size: 1.4em;
            margin-bottom: 10px;
        }
        .summary-text {
            font-size: 1.1em;
            color: #2c3e50;
            line-height: 1.8;
        }
        .insights-list {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .recommendations {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
        }
        .priority-high {
            background: #ffe6e6;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .priority-medium {
            background: #fff8e1;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .visualization {
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .ai-badge {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Executive Business Intelligence Report</h1>
            <div class="subtitle">
                Generated {{ generated_at }}
                {% if ai_powered %}<span class="ai-badge">🤖 AI-Powered Insights</span>{% endif %}
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-text">
                {{ executive_summary.summary_text or "Executive summary not available" }}
            </div>
            
            {% if executive_summary.key_findings %}
            <h3>Key Findings</h3>
            <div class="insights-list">
                {% for finding in executive_summary.key_findings %}
                <li>{{ finding }}</li>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <!-- Business Insights -->
        <div class="section">
            <h2>🔍 Business Insights</h2>
            <div class="summary-text">
                {{ business_insights.insights_narrative or "Business insights not available" }}
            </div>
            
            {% if business_insights.key_insights %}
            <h3>Strategic Insights</h3>
            <div class="insights-list">
                {% for insight in business_insights.key_insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <!-- Visualizations -->
        {% if visualizations %}
        <div class="section">
            <h2>📈 Data Visualizations</h2>
            {% for viz in visualizations %}
            <div class="visualization">
                <h3>{{ viz.title }}</h3>
                {% if viz.html %}
                {{ viz.html | safe }}
                {% else %}
                <p>Visualization not available: {{ viz.get('error', 'Unknown error') }}</p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Strategic Recommendations -->
        <div class="section">
            <h2>🎯 Strategic Recommendations</h2>
            <div class="summary-text">
                {{ strategic_recommendations.recommendations_narrative or "Recommendations not available" }}
            </div>
            
            {% if strategic_recommendations.recommendations %}
            <h3>Action Items</h3>
            {% for rec in strategic_recommendations.recommendations %}
            <div class="priority-high">
                <strong>{{ rec.title or "Recommendation" }}:</strong> {{ rec.description or rec }}
            </div>
            {% endfor %}
            {% endif %}
            
            {% if strategic_recommendations.priority_actions %}
            <h3>Priority Actions</h3>
            {% for action in strategic_recommendations.priority_actions %}
            <div class="priority-medium">
                {{ action }}
            </div>
            {% endfor %}
            {% endif %}
        </div>

        <div class="footer">
            <p>Report generated using advanced data analytics and AI-powered insights | {{ generated_at }}</p>
            <p>Data sources: {{ data_sources | join(', ') if data_sources else 'Internal analysis' }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        
        # Prepare template variables
        template_vars = {
            'generated_at': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            'ai_powered': self.model is not None,
            'executive_summary': executive_summary,
            'business_insights': business_insights,
            'strategic_recommendations': strategic_recommendations,
            'visualizations': visualizations,
            'data_sources': [analysis_data.get('metadata', {}).get('source_file', 'Analysis Data')]
        }
        
        return template.render(**template_vars)

    # Fallback methods when Gemini is not available
    def _generate_fallback_executive_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic executive summary without AI."""
        metadata = analysis_data.get("metadata", {})
        basic_stats = analysis_data.get("basic_statistics", {})
        
        summary_text = f"""
This report analyzes {metadata.get('data_shape', [0, 0])[0]} records across {metadata.get('data_shape', [0, 0])[1]} variables. 
The analysis reveals key performance patterns and business opportunities for strategic decision-making.
Data quality indicators show {len(basic_stats.get('null_counts', {}))} variables with complete coverage.
        """
        
        return {
            "summary_text": summary_text.strip(),
            "key_findings": self._extract_key_findings(analysis_data),
            "generated_with_ai": False
        }

    def _generate_fallback_insights(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic insights without AI."""
        insights = analysis_data.get("narrative_insights", [])
        
        return {
            "insights_narrative": "Comprehensive analysis reveals important business patterns and opportunities for optimization.",
            "key_insights": insights[:5] if insights else ["Analysis completed successfully"],
            "generated_with_ai": False
        }

    def _generate_fallback_recommendations(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic recommendations without AI."""
        recommendations = analysis_data.get("business_recommendations", [])
        
        return {
            "recommendations_narrative": "Based on the analysis, several strategic opportunities have been identified for business improvement.",
            "recommendations": recommendations[:5] if recommendations else ["Review and monitor key performance indicators"],
            "generated_with_ai": False
        }

    # Helper methods
    def _extract_key_findings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis data."""
        findings = []
        
        try:
            # Add findings from various analysis sections
            narrative_insights = analysis_data.get("narrative_insights", [])
            findings.extend(narrative_insights[:3])
            
            # Add metric performance findings
            key_metrics = analysis_data.get("key_metrics", {})
            if key_metrics.get("metric_performance"):
                findings.extend(key_metrics["metric_performance"][:2])
            
            # Add contribution insights
            contribution = analysis_data.get("contribution_analysis", {})
            if contribution.get("key_insights"):
                findings.extend(contribution["key_insights"][:2])
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
        
        return findings[:5]  # Limit to top 5 findings

    def _extract_performance_indicators(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance indicators."""
        try:
            key_metrics = analysis_data.get("key_metrics", {})
            business_metrics = key_metrics.get("business_metrics", {})
            
            kpis = {}
            for metric, values in business_metrics.items():
                kpis[metric] = {
                    "value": values.get("total", 0),
                    "growth_rate": values.get("growth_rate", 0)
                }
            
            return kpis
        except:
            return {}

    def _extract_structured_insights(self, insights_text: str) -> List[str]:
        """Extract structured insights from AI-generated text."""
        # Simple extraction - look for bullet points or numbered items
        lines = insights_text.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 10))):
                cleaned = line.lstrip('-•0123456789. ').strip()
                if cleaned:
                    insights.append(cleaned)
        
        return insights[:5]  # Top 5 insights

    def _extract_structured_recommendations(self, recommendations_text: str) -> List[Dict[str, str]]:
        """Extract structured recommendations from AI-generated text."""
        lines = recommendations_text.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 10))):
                cleaned = line.lstrip('-•0123456789. ').strip()
                if cleaned:
                    recommendations.append({"description": cleaned})
        
        return recommendations[:5]  # Top 5 recommendations

    def _extract_priority_actions(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract priority actions from analysis."""
        actions = []
        
        try:
            # From business recommendations
            business_recs = analysis_data.get("business_recommendations", [])
            actions.extend(business_recs[:3])
            
            # Add momentum-based actions
            momentum = analysis_data.get("momentum_analysis", {})
            if momentum.get("momentum_trend") == "decreasing":
                actions.append("Address declining momentum in key metrics")
            
        except Exception as e:
            logger.error(f"Error extracting priority actions: {e}")
        
        return actions[:5]

    def _extract_market_observations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract market observations from analysis."""
        observations = []
        
        try:
            # From trend analysis
            trends = analysis_data.get("trend_analysis", {})
            if trends.get("seasonal_patterns", {}).get("has_seasonality"):
                observations.append("Seasonal patterns detected in performance metrics")
            
            # From outlier detection
            outliers = analysis_data.get("outlier_detection", {})
            if outliers.get("total_outlier_records", 0) > 0:
                observations.append(f"Significant outliers detected in {outliers.get('total_outlier_records', 0)} records")
            
        except Exception as e:
            logger.error(f"Error extracting market observations: {e}")
        
        return observations

    async def _create_basic_report(self, data_handle_id: str, report_type: str) -> Dict[str, Any]:
        """Create basic report for backward compatibility."""
        try:
            data = self.data_manager.get_data(data_handle_id)
            
            # Create basic visualization
            if isinstance(data, pd.DataFrame):
                fig = px.bar(data.head(10))
                report_content = fig.to_html()
            else:
                report_content = f"<h1>Basic Report</h1><p>Data processed: {type(data).__name__}</p>"
            
            report_handle = self.data_manager.create_handle(
                data=report_content,
                data_type="basic_report",
                metadata={
                    "original_handle": data_handle_id,
                    "report_type": report_type,
                }
            )
            
            return {
                "status": "completed",
                "report_data_handle_id": report_handle.handle_id,
            }
            
        except Exception as e:
            logger.exception(f"Error creating basic report: {e}")
            raise

# Maintain backward compatibility
PresentationAgentExecutor = EnhancedPresentationExecutor 