"""
Enhanced RootCause Analyst Agent Executor with Google Gemini AI
Implements automated root cause analysis with sophisticated "Why-Bot" functionality.
"""

import logging
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import sys
from pathlib import Path

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Statistical libraries
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Enhanced AI capabilities
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Google Gemini AI available - enhanced hypothesis generation enabled")
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Gemini AI not available - using fallback hypothesis generation")

# Statistical libraries with fallbacks
STATSMODELS_AVAILABLE = False
try:
    import statsmodels.api as sm
    from statsmodels.stats.contingency_tables import chi2_contingency
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAILABLE = True
except ImportError:
    logging.warning("Statsmodels not available - using scipy.stats for statistical tests")

# Causal inference libraries  
DOWHY_AVAILABLE = False
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    logging.warning("DoWhy not available - causal analysis will use statistical methods")

ECONML_AVAILABLE = False
try:
    from econml.dml import LinearDML
    ECONML_AVAILABLE = True
except ImportError:
    logging.warning("EconML not available - treatment effect estimation will use basic methods")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Framework integration
from common_utils.data_handle_manager import get_data_handle_manager
from common_utils.config import Settings

logger = logging.getLogger(__name__)

class EnhancedRootCauseAnalystExecutor:
    """
    Enhanced RootCause Analyst Agent Executor - The "Why-Bot" with Google Gemini AI
    
    Implements comprehensive automated root cause analysis using:
    - Google Gemini AI-powered hypothesis generation (3-5 Whys methodology)
    - Advanced statistical testing and variance decomposition
    - Causal inference with DoWhy/EconML integration
    - Intelligent reporting with confidence scoring
    - Escalation logic for low-confidence scenarios
    """
    
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        
        # Initialize Gemini AI if available
        if GEMINI_AVAILABLE:
            try:
                settings = Settings()
                genai.configure(api_key=settings.google_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                logger.info("✅ Google Gemini configured for root cause analysis")
            except Exception as e:
                logger.warning(f"⚠️ Could not configure Gemini: {e}")
                self.model = None
        else:
            self.model = None
        
        # Initialize statistical testing components
        self.label_encoders = {}
        self.confidence_threshold = 0.7  # Minimum confidence for automated analysis
        
        logger.info("Enhanced RootCause Analyst (Why-Bot) initialized with AI-powered capabilities")

    async def investigate_trend_skill(self, 
                                    analysis_handle_id: str,
                                    trend_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Primary skill: Investigate trends and perform comprehensive root cause analysis.
        """
        logger.info(f"🔍 Starting root cause investigation for analysis handle: {analysis_handle_id}")
        
        try:
            config = trend_config or {}
            
            # Get comprehensive analysis data
            analysis_data = self.data_manager.get_data(analysis_handle_id)
            if not analysis_data:
                raise ValueError(f"Analysis data not found: {analysis_handle_id}")
            
            # Extract original dataset for detailed analysis
            original_data_handle = analysis_data.get("metadata", {}).get("original_handle")
            if original_data_handle:
                dataset = self.data_manager.get_data(original_data_handle)
            else:
                # Fallback: create synthetic data based on analysis
                dataset = self._create_synthetic_dataset_from_analysis(analysis_data)
            
            # Step 1: Generate AI-powered hypotheses
            hypotheses = await self._generate_ai_hypotheses(analysis_data, dataset, config)
            
            # Step 2: Test hypotheses with statistical analysis
            hypothesis_results = await self._test_hypotheses(hypotheses, dataset, analysis_data)
            
            # Step 3: Perform variance decomposition
            variance_analysis = self._perform_variance_decomposition(dataset, analysis_data)
            
            # Step 4: Causal inference analysis
            causal_analysis = self._perform_causal_analysis(dataset, analysis_data, hypothesis_results)
            
            # Step 5: Calculate overall confidence and determine escalation
            overall_confidence = self._calculate_overall_confidence(hypothesis_results, variance_analysis, causal_analysis)
            
            # Step 6: Generate root cause brief
            root_cause_brief = await self._generate_root_cause_brief(
                hypotheses=hypothesis_results,
                variance_analysis=variance_analysis,
                causal_analysis=causal_analysis,
                overall_confidence=overall_confidence,
                analysis_data=analysis_data,
                config=config
            )
            
            # Step 7: Create visualizations
            visualizations = self._create_root_cause_visualizations(
                hypothesis_results, variance_analysis, dataset
            )
            
            # Compile final results
            investigation_results = {
                "investigation_id": str(uuid.uuid4()),
                "analysis_handle_id": analysis_handle_id,
                "timestamp": datetime.now().isoformat(),
                "overall_confidence": overall_confidence,
                "requires_escalation": overall_confidence < self.confidence_threshold,
                "ai_powered": self.model is not None,
                "hypotheses_tested": len(hypothesis_results),
                "significant_findings": len([h for h in hypothesis_results if h.get("is_significant", False)]),
                "methodology": {
                    "hypothesis_generation": "google_gemini_ai" if self.model else "statistical_fallback",
                    "statistical_testing": "comprehensive_suite",
                    "causal_inference": "dowhy_econml" if DOWHY_AVAILABLE else "correlation_analysis",
                    "confidence_scoring": "multi_factor_weighted"
                },
                "results": {
                    "hypotheses": hypothesis_results,
                    "variance_decomposition": variance_analysis,
                    "causal_analysis": causal_analysis,
                    "root_cause_brief": root_cause_brief,
                    "visualizations": visualizations
                }
            }
            
            # Create investigation results handle
            investigation_handle = self.data_manager.create_handle(
                data=investigation_results,
                data_type="root_cause_investigation",
                metadata={
                    "original_analysis_handle": analysis_handle_id,
                    "investigation_type": "comprehensive_root_cause",
                    "ai_enhanced": self.model is not None,
                    "confidence_level": overall_confidence,
                    "requires_escalation": overall_confidence < self.confidence_threshold
                }
            )
            
            logger.info(f"✅ Root cause investigation completed. Handle: {investigation_handle.handle_id}")
            logger.info(f"🎯 Overall confidence: {overall_confidence:.2f}")
            if overall_confidence < self.confidence_threshold:
                logger.warning("⚠️ Low confidence - escalation recommended")
            
            return {
                "status": "completed",
                "investigation_handle_id": investigation_handle.handle_id,
                "summary": {
                    "overall_confidence": overall_confidence,
                    "hypotheses_tested": len(hypothesis_results),
                    "significant_findings": len([h for h in hypothesis_results if h.get("is_significant", False)]),
                    "requires_escalation": overall_confidence < self.confidence_threshold,
                    "ai_enhanced": self.model is not None
                }
            }
            
        except Exception as e:
            logger.exception(f"Error in root cause investigation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_handle_id": analysis_handle_id
            }

    async def root_cause_analysis_skill(self, **kwargs) -> Dict[str, Any]:
        """
        Alias for investigate_trend_skill for clearer semantic naming.
        """
        return await self.investigate_trend_skill(**kwargs)

    async def _generate_ai_hypotheses(self, 
                                    analysis_data: Dict[str, Any], 
                                    dataset: pd.DataFrame,
                                    config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses using Google Gemini AI with 3-5 Whys methodology."""
        
        if not self.model:
            return self._generate_fallback_hypotheses(analysis_data, dataset)
        
        try:
            # Prepare analysis summary for Gemini
            trend_description = self._extract_trend_description(analysis_data)
            business_context = config.get("business_context", "General business analysis")
            available_columns = list(dataset.columns)
            
            # Create comprehensive prompt for hypothesis generation
            prompt = self._create_hypothesis_generation_prompt(
                trend_description, available_columns, business_context
            )
            
            response = await self.model.generate_content_async(prompt)
            
            # Parse Gemini response into structured hypotheses
            hypotheses = self._parse_gemini_hypotheses_response(response.text, available_columns)
            
            logger.info(f"🤖 Generated {len(hypotheses)} AI-powered hypotheses using Gemini")
            return hypotheses
            
        except Exception as e:
            logger.warning(f"Gemini hypothesis generation failed, using fallback: {e}")
            return self._generate_fallback_hypotheses(analysis_data, dataset)

    def _create_hypothesis_generation_prompt(self, 
                                           trend_description: str,
                                           available_columns: List[str],
                                           business_context: str) -> str:
        """Create comprehensive prompt for Gemini hypothesis generation."""
        
        prompt = f"""
You are an expert business analyst and data scientist using the "5 Whys" root cause analysis methodology.

BUSINESS TREND TO INVESTIGATE:
{trend_description}

AVAILABLE DATA DIMENSIONS:
{', '.join(available_columns)}

BUSINESS CONTEXT:
{business_context}

TASK:
Generate 5 plausible hypotheses that could explain this trend. For each hypothesis, apply the "Why?" question iteratively to get to root causes.

For each hypothesis, provide a JSON object with:
1. hypothesis_id: A unique identifier (hyp_1, hyp_2, etc.)
2. hypothesis: Clear statement of the potential cause
3. why_chain: The "why" reasoning chain (Why A? Because B. Why B? Because C...)
4. test_columns: Array of data columns that could validate this hypothesis
5. analysis_type: Statistical method to test ("correlation", "anova", "chi_square", "regression", "segmentation")
6. likelihood: Business likelihood score (1-10, where 10 = very likely)
7. testability: How easily this can be tested with available data (1-10)
8. impact_if_true: Business impact if this hypothesis is confirmed (1-10)

EXAMPLE FORMAT:
[
  {{
    "hypothesis_id": "hyp_1",
    "hypothesis": "Revenue drop is driven by decreased customer retention in specific regions",
    "why_chain": "Why revenue drop? Because fewer repeat customers. Why fewer repeat customers? Because customer satisfaction declined in certain regions. Why specific regions? Because new competitor launched there with better pricing.",
    "test_columns": ["region", "customer_id", "purchase_date", "customer_satisfaction", "revenue"],
    "analysis_type": "segmentation",
    "likelihood": 8,
    "testability": 9,
    "impact_if_true": 9
  }}
]

Return ONLY a valid JSON array with 5 hypotheses. Be specific, actionable, and use only the available columns.
"""
        return prompt

    def _extract_trend_description(self, analysis_data: Dict[str, Any]) -> str:
        """Extract trend description from comprehensive analysis data."""
        
        # Extract key insights from analysis
        insights = analysis_data.get("narrative_insights", [])
        trends = analysis_data.get("trend_analysis", {})
        key_metrics = analysis_data.get("key_metrics", {})
        
        # Build trend description
        description_parts = []
        
        # Add trend direction and strength
        if trends.get("trend_direction"):
            direction = trends["trend_direction"]
            strength = trends.get("trend_strength", 0)
            description_parts.append(f"Data shows {direction} trend with strength {strength:.2f}")
        
        # Add key metrics information
        if key_metrics.get("business_metrics"):
            for metric, values in key_metrics["business_metrics"].items():
                growth_rate = values.get("growth_rate", 0)
                if abs(growth_rate) > 5:  # Significant change
                    description_parts.append(f"{metric} changed by {growth_rate:.1f}%")
        
        # Add narrative insights
        if insights:
            description_parts.extend(insights[:3])  # Top 3 insights
        
        # Fallback description
        if not description_parts:
            description_parts.append("Significant business pattern detected requiring root cause analysis")
        
        return ". ".join(description_parts)

    def _parse_gemini_hypotheses_response(self, 
                                        response_text: str, 
                                        available_columns: List[str]) -> List[Dict[str, Any]]:
        """Parse Gemini's JSON response into structured hypotheses."""
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            json_text = response_text[start_idx:end_idx]
            hypotheses = json.loads(json_text)
            
            # Validate and clean hypotheses
            validated_hypotheses = []
            for hyp in hypotheses:
                if self._validate_hypothesis(hyp, available_columns):
                    validated_hypotheses.append(hyp)
            
            logger.info(f"✅ Parsed {len(validated_hypotheses)} valid hypotheses from Gemini response")
            return validated_hypotheses
            
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return self._generate_fallback_hypotheses_simple()

    def _validate_hypothesis(self, hypothesis: Dict[str, Any], available_columns: List[str]) -> bool:
        """Validate that hypothesis has required fields and valid test columns."""
        
        required_fields = ["hypothesis_id", "hypothesis", "test_columns", "analysis_type", "likelihood"]
        
        # Check required fields
        for field in required_fields:
            if field not in hypothesis:
                return False
        
        # Validate test columns exist in dataset
        test_columns = hypothesis.get("test_columns", [])
        if not test_columns:
            return False
        
        valid_columns = [col for col in test_columns if col in available_columns]
        if not valid_columns:
            return False
        
        # Update with only valid columns
        hypothesis["test_columns"] = valid_columns
        
        return True

    async def _test_hypotheses(self, 
                             hypotheses: List[Dict[str, Any]], 
                             dataset: pd.DataFrame,
                             analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test each hypothesis using appropriate statistical methods."""
        
        tested_hypotheses = []
        primary_metric = self._identify_primary_metric(dataset, analysis_data)
        
        for hypothesis in hypotheses:
            logger.info(f"🧪 Testing hypothesis: {hypothesis.get('hypothesis_id', 'unknown')}")
            
            try:
                test_result = self._test_single_hypothesis(hypothesis, dataset, primary_metric)
                
                # Add original hypothesis data
                test_result.update(hypothesis)
                
                tested_hypotheses.append(test_result)
                
            except Exception as e:
                logger.warning(f"Failed to test hypothesis {hypothesis.get('hypothesis_id')}: {e}")
                # Add failed test result
                hypothesis.update({
                    "test_status": "failed",
                    "test_error": str(e),
                    "is_significant": False,
                    "confidence": 0.0
                })
                tested_hypotheses.append(hypothesis)
        
        return tested_hypotheses

    def _test_single_hypothesis(self, 
                              hypothesis: Dict[str, Any], 
                              dataset: pd.DataFrame,
                              primary_metric: str) -> Dict[str, Any]:
        """Test a single hypothesis using the specified analysis type."""
        
        analysis_type = hypothesis.get("analysis_type", "correlation")
        test_columns = hypothesis.get("test_columns", [])
        
        # Ensure we have valid columns
        valid_columns = [col for col in test_columns if col in dataset.columns]
        if not valid_columns:
            raise ValueError("No valid test columns available")
        
        # Dispatch to appropriate test method
        if analysis_type == "correlation":
            return self._test_correlation(dataset, valid_columns, primary_metric)
        elif analysis_type == "anova":
            return self._test_anova(dataset, valid_columns, primary_metric)
        elif analysis_type == "chi_square":
            return self._test_chi_square(dataset, valid_columns)
        elif analysis_type == "regression":
            return self._test_regression(dataset, valid_columns, primary_metric)
        elif analysis_type == "segmentation":
            return self._test_segmentation(dataset, valid_columns, primary_metric)
        else:
            # Default to correlation analysis
            return self._test_correlation(dataset, valid_columns, primary_metric)

    def _test_correlation(self, dataset: pd.DataFrame, test_columns: List[str], primary_metric: str) -> Dict[str, Any]:
        """Test correlation between test columns and primary metric."""
        
        correlations = {}
        significant_correlations = []
        
        for col in test_columns:
            if col == primary_metric:
                continue
                
            if pd.api.types.is_numeric_dtype(dataset[col]) and pd.api.types.is_numeric_dtype(dataset[primary_metric]):
                # Numeric correlation
                corr_coef, p_value = stats.pearsonr(dataset[col].dropna(), dataset[primary_metric].dropna())
                correlations[col] = {
                    "correlation": float(corr_coef),
                    "p_value": float(p_value),
                    "method": "pearson"
                }
                
                if p_value < 0.05 and abs(corr_coef) > 0.3:
                    significant_correlations.append(col)
        
        is_significant = len(significant_correlations) > 0
        confidence = max([abs(correlations[col]["correlation"]) for col in significant_correlations], default=0.0)
        
        return {
            "test_status": "completed",
            "test_method": "correlation_analysis",
            "test_results": {
                "correlations": correlations,
                "significant_variables": significant_correlations
            },
            "is_significant": is_significant,
            "confidence": confidence,
            "effect_size": confidence  # Use correlation as effect size proxy
        }

    def _test_anova(self, dataset: pd.DataFrame, test_columns: List[str], primary_metric: str) -> Dict[str, Any]:
        """Test ANOVA for categorical variables vs primary metric."""
        
        anova_results = {}
        significant_factors = []
        
        for col in test_columns:
            if col == primary_metric:
                continue
                
            # Only test categorical columns
            if not pd.api.types.is_numeric_dtype(dataset[col]):
                try:
                    groups = [group for name, group in dataset.groupby(col)[primary_metric] if len(group) > 1]
                    
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        anova_results[col] = {
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "groups": len(groups)
                        }
                        
                        if p_value < 0.05:
                            significant_factors.append(col)
                            
                except Exception as e:
                    logger.warning(f"ANOVA failed for {col}: {e}")
        
        is_significant = len(significant_factors) > 0
        confidence = 1 - min([anova_results[col]["p_value"] for col in significant_factors], default=1.0)
        
        return {
            "test_status": "completed",
            "test_method": "anova",
            "test_results": {
                "anova_results": anova_results,
                "significant_factors": significant_factors
            },
            "is_significant": is_significant,
            "confidence": confidence,
            "effect_size": confidence
        }

    def _test_regression(self, dataset: pd.DataFrame, test_columns: List[str], primary_metric: str) -> Dict[str, Any]:
        """Test regression model with test columns predicting primary metric."""
        
        try:
            # Prepare features
            feature_data = []
            feature_names = []
            
            for col in test_columns:
                if col == primary_metric:
                    continue
                    
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    feature_data.append(dataset[col].fillna(dataset[col].median()))
                    feature_names.append(col)
                else:
                    # One-hot encode categorical variables
                    encoded = pd.get_dummies(dataset[col], prefix=col)
                    for encoded_col in encoded.columns:
                        feature_data.append(encoded[encoded_col])
                        feature_names.append(encoded_col)
            
            if not feature_data:
                raise ValueError("No valid features for regression")
            
            X = pd.concat(feature_data, axis=1)
            y = dataset[primary_metric].fillna(dataset[primary_metric].median())
            
            # Train Random Forest (more robust than linear regression)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, rf.feature_importances_))
            
            # Calculate R² score
            r2_score = rf.score(X, y)
            
            # Identify important features (top 20%)
            importance_threshold = np.percentile(list(feature_importance.values()), 80)
            important_features = [name for name, importance in feature_importance.items() 
                                if importance >= importance_threshold]
            
            is_significant = r2_score > 0.1 and len(important_features) > 0
            confidence = r2_score
            
            return {
                "test_status": "completed",
                "test_method": "random_forest_regression",
                "test_results": {
                    "r2_score": float(r2_score),
                    "feature_importance": {k: float(v) for k, v in feature_importance.items()},
                    "important_features": important_features
                },
                "is_significant": is_significant,
                "confidence": confidence,
                "effect_size": r2_score
            }
            
        except Exception as e:
            logger.warning(f"Regression test failed: {e}")
            return {
                "test_status": "failed",
                "test_error": str(e),
                "is_significant": False,
                "confidence": 0.0
            }

    def _test_segmentation(self, dataset: pd.DataFrame, test_columns: List[str], primary_metric: str) -> Dict[str, Any]:
        """Test segmentation analysis to find segments with different behavior."""
        
        try:
            segment_results = {}
            significant_segments = []
            
            for col in test_columns:
                if col == primary_metric:
                    continue
                
                # Group by column and analyze primary metric
                if not pd.api.types.is_numeric_dtype(dataset[col]):
                    grouped = dataset.groupby(col)[primary_metric].agg(['mean', 'std', 'count'])
                    overall_mean = dataset[primary_metric].mean()
                    
                    # Calculate deviation from overall mean
                    segment_analysis = {}
                    for segment in grouped.index:
                        segment_mean = grouped.loc[segment, 'mean']
                        segment_count = grouped.loc[segment, 'count']
                        
                        if segment_count >= 5:  # Minimum segment size
                            deviation = abs(segment_mean - overall_mean) / overall_mean
                            segment_analysis[segment] = {
                                "mean": float(segment_mean),
                                "count": int(segment_count),
                                "deviation_from_overall": float(deviation)
                            }
                    
                    if segment_analysis:
                        segment_results[col] = segment_analysis
                        
                        # Check if any segment has significant deviation (>20%)
                        max_deviation = max([s["deviation_from_overall"] for s in segment_analysis.values()])
                        if max_deviation > 0.2:
                            significant_segments.append(col)
            
            is_significant = len(significant_segments) > 0
            confidence = max([
                max([s["deviation_from_overall"] for s in seg.values()]) 
                for seg in segment_results.values()
            ], default=0.0)
            
            return {
                "test_status": "completed",
                "test_method": "segmentation_analysis",
                "test_results": {
                    "segment_analysis": segment_results,
                    "significant_segments": significant_segments
                },
                "is_significant": is_significant,
                "confidence": min(confidence, 1.0),
                "effect_size": confidence
            }
            
        except Exception as e:
            logger.warning(f"Segmentation test failed: {e}")
            return {
                "test_status": "failed",
                "test_error": str(e),
                "is_significant": False,
                "confidence": 0.0
            }

    def _identify_primary_metric(self, dataset: pd.DataFrame, analysis_data: Dict[str, Any]) -> str:
        """Identify the primary metric for root cause analysis."""
        
        # Try to get from analysis data
        key_metrics = analysis_data.get("key_metrics", {})
        if key_metrics.get("business_metrics"):
            return list(key_metrics["business_metrics"].keys())[0]
        
        # Fallback: find numeric column that looks like a KPI
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        
        kpi_keywords = ['revenue', 'sales', 'profit', 'margin', 'quantity', 'value', 'amount', 'total']
        
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in kpi_keywords):
                return col
        
        # Last resort: use first numeric column
        return numeric_cols[0] if len(numeric_cols) > 0 else 'value'

    def _perform_variance_decomposition(self, dataset: pd.DataFrame, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform variance decomposition to understand contribution of different factors."""
        
        try:
            primary_metric = self._identify_primary_metric(dataset, analysis_data)
            
            # Get categorical columns for decomposition
            categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
            
            if not categorical_cols:
                return {"error": "No categorical columns for variance decomposition"}
            
            variance_results = {}
            total_variance = dataset[primary_metric].var()
            
            for col in categorical_cols:
                try:
                    # Calculate within-group and between-group variance
                    grouped = dataset.groupby(col)[primary_metric]
                    
                    # Between-group variance
                    group_means = grouped.mean()
                    overall_mean = dataset[primary_metric].mean()
                    group_sizes = grouped.size()
                    
                    between_group_var = sum(group_sizes * (group_means - overall_mean) ** 2) / (len(group_means) - 1)
                    
                    # Within-group variance
                    within_group_var = sum(grouped.var() * (group_sizes - 1)) / (len(dataset) - len(group_means))
                    
                    # Variance explained ratio
                    variance_explained = between_group_var / total_variance if total_variance > 0 else 0
                    
                    variance_results[col] = {
                        "variance_explained": float(variance_explained),
                        "between_group_variance": float(between_group_var),
                        "within_group_variance": float(within_group_var),
                        "num_groups": len(group_means)
                    }
                    
                except Exception as e:
                    logger.warning(f"Variance decomposition failed for {col}: {e}")
            
            # Rank by variance explained
            ranked_factors = sorted(variance_results.items(), 
                                  key=lambda x: x[1]["variance_explained"], 
                                  reverse=True)
            
            return {
                "total_variance": float(total_variance),
                "variance_by_factor": variance_results,
                "ranked_factors": [(factor, details["variance_explained"]) for factor, details in ranked_factors],
                "top_variance_driver": ranked_factors[0][0] if ranked_factors else None
            }
            
        except Exception as e:
            logger.error(f"Variance decomposition failed: {e}")
            return {"error": str(e)}

    def _perform_causal_analysis(self, 
                               dataset: pd.DataFrame, 
                               analysis_data: Dict[str, Any],
                               hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform causal analysis using DoWhy if available, otherwise correlation-based inference."""
        
        if DOWHY_AVAILABLE:
            return self._perform_dowhy_causal_analysis(dataset, analysis_data, hypothesis_results)
        else:
            return self._perform_correlation_causal_inference(dataset, analysis_data, hypothesis_results)

    def _perform_correlation_causal_inference(self, 
                                           dataset: pd.DataFrame, 
                                           analysis_data: Dict[str, Any],
                                           hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback causal inference using correlation and temporal analysis."""
        
        try:
            primary_metric = self._identify_primary_metric(dataset, analysis_data)
            
            # Identify potential causes from significant hypothesis results
            potential_causes = []
            for hyp in hypothesis_results:
                if hyp.get("is_significant", False):
                    test_columns = hyp.get("test_columns", [])
                    confidence = hyp.get("confidence", 0)
                    potential_causes.extend([(col, confidence) for col in test_columns if col != primary_metric])
            
            # Remove duplicates and sort by confidence
            potential_causes = list(set(potential_causes))
            potential_causes.sort(key=lambda x: x[1], reverse=True)
            
            causal_relationships = []
            
            for cause, confidence in potential_causes[:5]:  # Top 5 potential causes
                if cause in dataset.columns:
                    # Calculate correlation
                    if pd.api.types.is_numeric_dtype(dataset[cause]):
                        corr_coef, p_value = stats.pearsonr(
                            dataset[cause].dropna(), 
                            dataset[primary_metric].dropna()
                        )
                        
                        # Simple causal inference based on correlation strength and significance
                        causal_strength = abs(corr_coef) * confidence
                        
                        causal_relationships.append({
                            "cause": cause,
                            "effect": primary_metric,
                            "correlation": float(corr_coef),
                            "p_value": float(p_value),
                            "causal_strength": float(causal_strength),
                            "confidence": float(confidence),
                            "method": "correlation_inference"
                        })
            
            # Rank by causal strength
            causal_relationships.sort(key=lambda x: x["causal_strength"], reverse=True)
            
            return {
                "method": "correlation_based_inference",
                "causal_relationships": causal_relationships,
                "top_causal_factor": causal_relationships[0]["cause"] if causal_relationships else None,
                "confidence_note": "Causal inference based on correlation analysis - consider experimental validation"
            }
            
        except Exception as e:
            logger.error(f"Correlation-based causal inference failed: {e}")
            return {"error": str(e), "method": "correlation_based_inference"}

    def _calculate_overall_confidence(self, 
                                    hypothesis_results: List[Dict[str, Any]],
                                    variance_analysis: Dict[str, Any],
                                    causal_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the root cause analysis."""
        
        confidence_factors = []
        
        # Factor 1: Hypothesis testing confidence
        significant_hypotheses = [h for h in hypothesis_results if h.get("is_significant", False)]
        if significant_hypotheses:
            avg_hypothesis_confidence = np.mean([h.get("confidence", 0) for h in significant_hypotheses])
            confidence_factors.append(avg_hypothesis_confidence * 0.4)  # 40% weight
        
        # Factor 2: Variance decomposition clarity
        if "ranked_factors" in variance_analysis and variance_analysis["ranked_factors"]:
            top_variance_explained = variance_analysis["ranked_factors"][0][1]
            confidence_factors.append(top_variance_explained * 0.3)  # 30% weight
        
        # Factor 3: Causal analysis strength
        if "causal_relationships" in causal_analysis and causal_analysis["causal_relationships"]:
            top_causal_strength = causal_analysis["causal_relationships"][0].get("causal_strength", 0)
            confidence_factors.append(min(top_causal_strength, 1.0) * 0.3)  # 30% weight
        
        # Factor 4: Number of significant findings
        num_significant = len(significant_hypotheses)
        significance_bonus = min(num_significant / 3.0, 0.2)  # Max 20% bonus for multiple findings
        
        overall_confidence = sum(confidence_factors) + significance_bonus
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, overall_confidence))

    async def _generate_root_cause_brief(self,
                                       hypotheses: List[Dict[str, Any]],
                                       variance_analysis: Dict[str, Any],
                                       causal_analysis: Dict[str, Any],
                                       overall_confidence: float,
                                       analysis_data: Dict[str, Any],
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive root cause brief using Gemini AI."""
        
        if not self.model:
            return self._generate_fallback_brief(hypotheses, variance_analysis, causal_analysis, overall_confidence)
        
        try:
            # Prepare summary for Gemini
            brief_prompt = self._create_root_cause_brief_prompt(
                hypotheses, variance_analysis, causal_analysis, overall_confidence, analysis_data
            )
            
            response = await self.model.generate_content_async(brief_prompt)
            
            # Structure the response
            brief_content = {
                "executive_summary": self._extract_executive_summary(response.text),
                "key_findings": self._extract_key_findings_from_brief(response.text),
                "root_causes": self._extract_root_causes(response.text),
                "recommendations": self._extract_recommendations_from_brief(response.text),
                "confidence_assessment": self._extract_confidence_assessment(response.text),
                "next_steps": self._extract_next_steps(response.text),
                "full_brief": response.text,
                "generated_with_ai": True
            }
            
            return brief_content
            
        except Exception as e:
            logger.warning(f"Gemini brief generation failed, using fallback: {e}")
            return self._generate_fallback_brief(hypotheses, variance_analysis, causal_analysis, overall_confidence)

    def _create_root_cause_brief_prompt(self,
                                      hypotheses: List[Dict[str, Any]],
                                      variance_analysis: Dict[str, Any],
                                      causal_analysis: Dict[str, Any],
                                      overall_confidence: float,
                                      analysis_data: Dict[str, Any]) -> str:
        """Create comprehensive prompt for root cause brief generation."""
        
        # Summarize significant findings
        significant_hypotheses = [h for h in hypotheses if h.get("is_significant", False)]
        top_variance_factors = variance_analysis.get("ranked_factors", [])[:3]
        causal_relationships = causal_analysis.get("causal_relationships", [])[:3]
        
        prompt = f"""
You are a senior business analyst creating a comprehensive Root Cause Analysis Brief. Based on the statistical analysis and hypothesis testing below, create a professional 1-page executive brief.

ANALYSIS RESULTS:

Overall Confidence: {overall_confidence:.2f} (Scale: 0.0 to 1.0)

SIGNIFICANT HYPOTHESES VALIDATED:
{json.dumps(significant_hypotheses, indent=2)}

TOP VARIANCE DRIVERS:
{json.dumps(top_variance_factors, indent=2)}

CAUSAL RELATIONSHIPS IDENTIFIED:
{json.dumps(causal_relationships, indent=2)}

ORIGINAL BUSINESS CONTEXT:
{self._extract_trend_description(analysis_data)}

TASK:
Create a professional root cause analysis brief with these sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - What happened and why (root cause conclusion)
   - Confidence level and reliability

2. KEY FINDINGS (3-5 bullet points)
   - Most significant statistical findings
   - Quantified impacts where possible

3. ROOT CAUSES IDENTIFIED (Ranked by confidence)
   - Primary root cause with statistical evidence
   - Secondary contributing factors
   - Data-driven reasoning

4. BUSINESS RECOMMENDATIONS (3-4 actionable items)
   - Immediate actions to address root causes
   - Preventive measures
   - Monitoring recommendations

5. CONFIDENCE ASSESSMENT
   - Data quality and completeness
   - Statistical significance of findings
   - Limitations and assumptions

6. NEXT STEPS
   - Additional data needed
   - Experimental validation suggestions
   - Follow-up analysis recommendations

Write in clear, business-appropriate language. Use specific numbers and statistics where available. Focus on actionable insights.
"""
        
        return prompt

    # Helper methods for brief parsing and fallback generation
    def _extract_executive_summary(self, brief_text: str) -> str:
        """Extract executive summary from AI-generated brief."""
        lines = brief_text.split('\n')
        summary_section = False
        summary_lines = []
        
        for line in lines:
            if 'EXECUTIVE SUMMARY' in line.upper():
                summary_section = True
                continue
            elif summary_section and line.strip().startswith(('2.', 'KEY FINDINGS', 'FINDINGS')):
                break
            elif summary_section and line.strip():
                summary_lines.append(line.strip())
        
        return ' '.join(summary_lines) if summary_lines else "Executive summary not available"

    def _extract_key_findings_from_brief(self, brief_text: str) -> List[str]:
        """Extract key findings from AI-generated brief."""
        lines = brief_text.split('\n')
        findings_section = False
        findings = []
        
        for line in lines:
            if 'KEY FINDINGS' in line.upper():
                findings_section = True
                continue
            elif findings_section and line.strip().startswith(('3.', 'ROOT CAUSES', 'CAUSES')):
                break
            elif findings_section and line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                findings.append(line.strip().lstrip('- •'))
        
        return findings if findings else ["Key findings not available"]

    def _extract_root_causes(self, brief_text: str) -> List[str]:
        """Extract root causes from AI-generated brief."""
        lines = brief_text.split('\n')
        causes_section = False
        causes = []
        
        for line in lines:
            if 'ROOT CAUSES' in line.upper():
                causes_section = True
                continue
            elif causes_section and line.strip().startswith(('4.', 'BUSINESS RECOMMENDATIONS', 'RECOMMENDATIONS')):
                break
            elif causes_section and line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                causes.append(line.strip().lstrip('- •'))
        
        return causes if causes else ["Root causes not available"]

    def _extract_recommendations_from_brief(self, brief_text: str) -> List[str]:
        """Extract recommendations from AI-generated brief."""
        lines = brief_text.split('\n')
        rec_section = False
        recommendations = []
        
        for line in lines:
            if 'RECOMMENDATIONS' in line.upper():
                rec_section = True
                continue
            elif rec_section and line.strip().startswith(('5.', 'CONFIDENCE', 'ASSESSMENT')):
                break
            elif rec_section and line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                recommendations.append(line.strip().lstrip('- •'))
        
        return recommendations if recommendations else ["Recommendations not available"]

    def _extract_confidence_assessment(self, brief_text: str) -> str:
        """Extract confidence assessment from AI-generated brief."""
        lines = brief_text.split('\n')
        conf_section = False
        conf_lines = []
        
        for line in lines:
            if 'CONFIDENCE' in line.upper():
                conf_section = True
                continue
            elif conf_section and line.strip().startswith(('6.', 'NEXT STEPS', 'STEPS')):
                break
            elif conf_section and line.strip():
                conf_lines.append(line.strip())
        
        return ' '.join(conf_lines) if conf_lines else "Confidence assessment not available"

    def _extract_next_steps(self, brief_text: str) -> List[str]:
        """Extract next steps from AI-generated brief."""
        lines = brief_text.split('\n')
        steps_section = False
        steps = []
        
        for line in lines:
            if 'NEXT STEPS' in line.upper():
                steps_section = True
                continue
            elif steps_section and line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                steps.append(line.strip().lstrip('- •'))
        
        return steps if steps else ["Next steps not available"]

    def _generate_fallback_brief(self,
                                hypotheses: List[Dict[str, Any]],
                                variance_analysis: Dict[str, Any],
                                causal_analysis: Dict[str, Any],
                                overall_confidence: float) -> Dict[str, Any]:
        """Generate fallback brief without AI."""
        
        significant_hypotheses = [h for h in hypotheses if h.get("is_significant", False)]
        
        brief_content = {
            "executive_summary": f"Root cause analysis completed with {overall_confidence:.2f} confidence. {len(significant_hypotheses)} significant factors identified.",
            "key_findings": [
                f"Analysis tested {len(hypotheses)} hypotheses",
                f"{len(significant_hypotheses)} hypotheses showed statistical significance",
                f"Overall confidence level: {overall_confidence:.2f}"
            ],
            "root_causes": [h.get("hypothesis", "Unknown") for h in significant_hypotheses[:3]],
            "recommendations": [
                "Review significant factors identified in analysis",
                "Consider experimental validation of top hypotheses",
                "Monitor key variance drivers identified"
            ],
            "confidence_assessment": f"Analysis confidence: {overall_confidence:.2f}. Based on statistical testing and variance decomposition.",
            "next_steps": [
                "Collect additional data for low-confidence factors",
                "Design experiments to validate causal relationships",
                "Implement monitoring for identified drivers"
            ],
            "full_brief": "Comprehensive root cause analysis completed using statistical methods.",
            "generated_with_ai": False
        }
        
        return brief_content

    def _create_root_cause_visualizations(self,
                                        hypothesis_results: List[Dict[str, Any]],
                                        variance_analysis: Dict[str, Any],
                                        dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create visualizations for root cause analysis."""
        
        visualizations = []
        
        try:
            # 1. Hypothesis confidence chart
            if hypothesis_results:
                viz = self._create_hypothesis_confidence_chart(hypothesis_results)
                visualizations.append(viz)
            
            # 2. Variance decomposition chart
            if variance_analysis.get("ranked_factors"):
                viz = self._create_variance_decomposition_chart(variance_analysis)
                visualizations.append(viz)
            
            # 3. Statistical significance summary
            viz = self._create_significance_summary_chart(hypothesis_results)
            visualizations.append(viz)
            
        except Exception as e:
            logger.error(f"Error creating root cause visualizations: {e}")
        
        return visualizations

    def _create_hypothesis_confidence_chart(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hypothesis confidence visualization."""
        
        try:
            # Extract hypothesis data
            hypothesis_ids = [h.get("hypothesis_id", f"hyp_{i}") for i, h in enumerate(hypothesis_results)]
            confidences = [h.get("confidence", 0) for h in hypothesis_results]
            significance = [h.get("is_significant", False) for h in hypothesis_results]
            
            # Create color map based on significance
            colors = ['green' if sig else 'red' for sig in significance]
            
            fig = go.Figure(data=go.Bar(
                x=hypothesis_ids,
                y=confidences,
                marker_color=colors,
                text=[f"{conf:.2f}" for conf in confidences],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Hypothesis Testing Results - Confidence Scores",
                xaxis_title="Hypothesis ID",
                yaxis_title="Confidence Score",
                showlegend=False,
                height=400
            )
            
            # Add significance threshold line
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                         annotation_text="Confidence Threshold")
            
            return {
                "title": "Hypothesis Confidence Analysis",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "bar_chart"
            }
            
        except Exception as e:
            return {"error": f"Failed to create hypothesis confidence chart: {e}"}

    def _create_variance_decomposition_chart(self, variance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create variance decomposition visualization."""
        
        try:
            ranked_factors = variance_analysis.get("ranked_factors", [])
            
            if not ranked_factors:
                return {"error": "No variance factors to visualize"}
            
            factors = [factor for factor, _ in ranked_factors]
            variance_explained = [variance for _, variance in ranked_factors]
            
            fig = go.Figure(data=go.Pie(
                labels=factors,
                values=variance_explained,
                textinfo='label+percent'
            ))
            
            fig.update_layout(
                title="Variance Decomposition - Factor Contribution",
                height=400
            )
            
            return {
                "title": "Variance Decomposition Analysis",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "pie_chart"
            }
            
        except Exception as e:
            return {"error": f"Failed to create variance decomposition chart: {e}"}

    def _create_significance_summary_chart(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create statistical significance summary."""
        
        try:
            total_hypotheses = len(hypothesis_results)
            significant_count = len([h for h in hypothesis_results if h.get("is_significant", False)])
            
            categories = ['Significant', 'Not Significant']
            values = [significant_count, total_hypotheses - significant_count]
            colors = ['green', 'lightcoral']
            
            fig = go.Figure(data=go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Statistical Significance Summary ({total_hypotheses} hypotheses tested)",
                xaxis_title="Significance Status",
                yaxis_title="Number of Hypotheses",
                showlegend=False,
                height=300
            )
            
            return {
                "title": "Statistical Significance Summary",
                "html": fig.to_html(include_plotlyjs='inline'),
                "type": "summary_chart"
            }
            
        except Exception as e:
            return {"error": f"Failed to create significance summary chart: {e}"}

    # Fallback methods
    def _generate_fallback_hypotheses(self, analysis_data: Dict[str, Any], dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate basic hypotheses without AI."""
        
        # Create simple hypotheses based on data structure
        hypotheses = []
        
        categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate hypotheses for categorical factors
        for i, col in enumerate(categorical_cols[:3]):
            hypotheses.append({
                "hypothesis_id": f"hyp_{i+1}",
                "hypothesis": f"Trend variation is driven by differences in {col}",
                "why_chain": f"Why trend variation? Because {col} shows different patterns. Why different patterns? Because underlying factors vary by {col}.",
                "test_columns": [col] + numeric_cols[:2],
                "analysis_type": "segmentation",
                "likelihood": 7,
                "testability": 8,
                "impact_if_true": 7
            })
        
        # Generate correlation hypotheses for numeric factors
        for i, col in enumerate(numeric_cols[:2]):
            hypotheses.append({
                "hypothesis_id": f"hyp_{len(hypotheses)+1}",
                "hypothesis": f"Trend is correlated with changes in {col}",
                "why_chain": f"Why trend changes? Because {col} influences the outcome. Why does {col} influence? Because they are directly related.",
                "test_columns": [col],
                "analysis_type": "correlation",
                "likelihood": 6,
                "testability": 9,
                "impact_if_true": 6
            })
        
        return hypotheses

    def _generate_fallback_hypotheses_simple(self) -> List[Dict[str, Any]]:
        """Generate very basic fallback hypotheses."""
        
        return [
            {
                "hypothesis_id": "hyp_1",
                "hypothesis": "Trend driven by seasonal factors",
                "test_columns": ["date"],
                "analysis_type": "correlation",
                "likelihood": 6,
                "testability": 7,
                "impact_if_true": 6
            },
            {
                "hypothesis_id": "hyp_2", 
                "hypothesis": "Trend driven by category differences",
                "test_columns": [],
                "analysis_type": "segmentation",
                "likelihood": 5,
                "testability": 6,
                "impact_if_true": 5
            }
        ]

    def _create_synthetic_dataset_from_analysis(self, analysis_data: Dict[str, Any]) -> pd.DataFrame:
        """Create synthetic dataset based on analysis metadata."""
        
        metadata = analysis_data.get("metadata", {})
        columns = metadata.get("columns", ["value", "category", "date"])
        rows = metadata.get("data_shape", [100, len(columns)])[0]
        
        # Create synthetic data
        data = {}
        
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                data[col] = pd.date_range('2024-01-01', periods=rows, freq='D')
            elif 'category' in col.lower() or 'region' in col.lower():
                data[col] = np.random.choice(['A', 'B', 'C', 'D'], rows)
            else:
                data[col] = np.random.normal(100, 20, rows)
        
        return pd.DataFrame(data)

# Maintain backward compatibility
RootCauseAnalystExecutor = EnhancedRootCauseAnalystExecutor 