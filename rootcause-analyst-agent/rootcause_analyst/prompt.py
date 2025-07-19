"""
Prompt templates for RootCause Analyst Agent (Why-Bot)
LLM prompts for hypothesis generation and causal reasoning.
"""

from langchain.prompts import PromptTemplate

# Main hypothesis generation prompt using 3-5 Whys methodology
HYPOTHESIS_GENERATION_PROMPT = PromptTemplate(
    input_variables=["trend_description", "columns", "business_context", "why_depth"],
    template="""
You are an expert business analyst and data scientist using the "5 Whys" root cause analysis methodology.

BUSINESS TREND TO INVESTIGATE:
{trend_description}

AVAILABLE DATA DIMENSIONS:
{columns}

BUSINESS CONTEXT:
{business_context}

TASK:
Generate {why_depth} plausible hypotheses that could explain this trend. For each hypothesis, apply the "Why?" question iteratively to get to root causes.

For each hypothesis, provide:
1. hypothesis_id: A unique identifier (hyp_1, hyp_2, etc.)
2. hypothesis: Clear statement of the potential cause
3. why_chain: The "why" reasoning chain (Why A? Because B. Why B? Because C...)
4. test_columns: Which data columns could validate this hypothesis
5. analysis_type: Statistical method to test (chi2, anova, correlation, regression)
6. likelihood: Business likelihood score (1-10, where 10 = very likely)
7. testability: How easily this can be tested with available data (1-10)
8. impact_if_true: Business impact if this hypothesis is confirmed (1-10)

EXAMPLE FORMAT:
[
  {{
    "hypothesis_id": "hyp_1",
    "hypothesis": "Revenue drop is driven by decreased customer retention in the West region",
    "why_chain": "Why revenue drop? Because fewer repeat customers. Why fewer repeat customers? Because customer satisfaction declined in West region. Why West region specifically? Because new competitor launched there.",
    "test_columns": ["region", "customer_id", "purchase_date", "customer_satisfaction"],
    "analysis_type": "segmentation_and_cohort_analysis",
    "likelihood": 8,
    "testability": 9,
    "impact_if_true": 9
  }}
]

Return ONLY valid JSON array. Be specific and actionable.
"""
)

# Causal inference prompt for interpreting statistical results
CAUSAL_INTERPRETATION_PROMPT = PromptTemplate(
    input_variables=["statistical_results", "business_context", "hypotheses"],
    template="""
You are a causal inference expert interpreting statistical analysis results.

STATISTICAL FINDINGS:
{statistical_results}

BUSINESS CONTEXT:
{business_context}

ORIGINAL HYPOTHESES:
{hypotheses}

TASK:
Interpret these statistical results from a causal perspective. For each significant finding:

1. Assess causality strength (correlation vs causation indicators)
2. Identify potential confounding variables
3. Suggest causal mechanisms
4. Rate confidence in causal claim (1-10)
5. Recommend next steps for causal validation

Consider Bradford Hill criteria:
- Strength of association
- Temporal sequence
- Dose-response relationship
- Biological plausibility (or business logic)
- Consistency across studies/segments

Format response as structured analysis with clear sections:
- CAUSAL ASSESSMENT
- CONFOUNDING RISKS  
- RECOMMENDED VALIDATION
- CONFIDENCE RATING
"""
)

# Business narrative generation prompt
NARRATIVE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["analysis_results", "confidence_score", "business_context"],
    template="""
You are a senior business consultant converting technical analysis into executive insights.

ANALYSIS RESULTS:
{analysis_results}

CONFIDENCE SCORE: {confidence_score}/10

BUSINESS CONTEXT:
{business_context}

TASK:
Create a clear, executive-level narrative explaining:

1. WHAT HAPPENED: The trend in business terms
2. WHY IT HAPPENED: Root causes identified (ranked by evidence strength)
3. WHAT IT MEANS: Business implications and risks/opportunities
4. WHAT TO DO: Prioritized action recommendations

REQUIREMENTS:
- Use business language, not statistical jargon
- Quantify impact where possible
- Be specific about next steps
- Acknowledge uncertainty appropriately
- Focus on actionable insights

TONE: Professional, confident but appropriately cautious about causal claims.
LENGTH: Executive summary suitable for 2-minute read.
"""
)

# Data quality assessment prompt
DATA_QUALITY_PROMPT = PromptTemplate(
    input_variables=["data_summary", "missing_patterns", "outlier_info"],
    template="""
Assess data quality for root cause analysis:

DATA SUMMARY:
{data_summary}

MISSING DATA PATTERNS:
{missing_patterns}

OUTLIER INFORMATION:
{outlier_info}

Evaluate:
1. Data sufficiency for causal inference
2. Missing data bias risks
3. Outlier impact on conclusions
4. Recommended data improvements

Rate overall data quality (1-10) and explain key limitations.
"""
)

# External context integration prompt
EXTERNAL_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["internal_findings", "external_events", "industry_context"],
    template="""
Integrate external context with internal root cause analysis:

INTERNAL ANALYSIS FINDINGS:
{internal_findings}

EXTERNAL EVENTS/NEWS:
{external_events}

INDUSTRY CONTEXT:
{industry_context}

TASK:
1. Identify external factors that could explain internal trends
2. Assess which internal findings might be spurious due to external causes
3. Suggest external data sources for validation
4. Update causal hypotheses incorporating external context

Focus on separating internal controllable factors from external market forces.
"""
)

# Experimental design prompt for validation
EXPERIMENT_DESIGN_PROMPT = PromptTemplate(
    input_variables=["causal_hypotheses", "available_levers", "constraints"],
    template="""
Design experiments to validate causal hypotheses:

CAUSAL HYPOTHESES TO TEST:
{causal_hypotheses}

AVAILABLE BUSINESS LEVERS:
{available_levers}

CONSTRAINTS:
{constraints}

For each testable hypothesis, design:
1. Experimental approach (A/B test, natural experiment, etc.)
2. Sample size requirements
3. Success metrics
4. Timeline and resource needs
5. Potential risks and mitigation

Prioritize experiments by:
- Business impact potential
- Feasibility
- Learning value
- Risk level

Provide specific, actionable experimental designs.
"""
) 