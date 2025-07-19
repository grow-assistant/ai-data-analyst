"""Agent card definitions for A2A system."""

from .types import AgentCard, AgentSkill, AgentCapabilities
from .config import settings

# Data Analysis Agent Cards
DATA_LOADER_AGENT_CARD = AgentCard(
    name="data_loader_agent",
    description="Agent for loading datasets from various sources including CSV, JSON, Tableau, and Excel files",
    url=f"http://localhost:{settings.data_loader_port}",
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="load_dataset",
            name="Load Dataset",
            description="Loads a dataset from file path and returns a data handle",
            tags=["data", "loading", "csv", "json", "excel"]
        ),
        AgentSkill(
            id="load_csv_file",
            name="Load CSV File",
            description="Specifically loads CSV files with custom parsing options",
            tags=["data", "csv", "loading"]
        ),
        AgentSkill(
            id="load_json_file",
            name="Load JSON File",
            description="Loads JSON data files and converts to tabular format",
            tags=["data", "json", "loading"]
        ),
        AgentSkill(
            id="extract_data_schema",
            name="Extract Data Schema",
            description="Extracts and returns metadata and schema information from datasets",
            tags=["data", "metadata", "schema"]
        )
    ]
)

DATA_CLEANING_AGENT_CARD = AgentCard(
    name="data_cleaning_agent",
    description="Agent for cleaning and preprocessing datasets to ensure data quality",
    url=f"http://localhost:10008",  # New port for cleaning agent
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="clean_dataset",
            name="Clean Dataset",
            description="Performs comprehensive data cleaning on a dataset using data handle",
            tags=["cleaning", "preprocessing", "quality"]
        ),
        AgentSkill(
            id="handle_missing_values",
            name="Handle Missing Values",
            description="Identifies and handles missing values using various strategies",
            tags=["cleaning", "missing-data", "imputation"]
        ),
        AgentSkill(
            id="remove_outliers",
            name="Remove Outliers",
            description="Detects and removes statistical outliers from dataset",
            tags=["cleaning", "outliers", "statistics"]
        ),
        AgentSkill(
            id="standardize_formats",
            name="Standardize Formats",
            description="Standardizes date formats, column names, and data types",
            tags=["cleaning", "standardization", "formatting"]
        )
    ]
)

DATA_ENRICHMENT_AGENT_CARD = AgentCard(
    name="data_enrichment_agent",
    description="Agent for enriching datasets with external context and derived features",
    url=f"http://localhost:10009",  # New port for enrichment agent
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="enrich_dataset",
            name="Enrich Dataset",
            description="Enriches a dataset with external data and derived features",
            tags=["enrichment", "context", "features"]
        ),
        AgentSkill(
            id="fetch_external_data",
            name="Fetch External Data",
            description="Retrieves data from external APIs and sources for enrichment",
            tags=["enrichment", "external-data", "apis"]
        ),
        AgentSkill(
            id="calculate_derived_features",
            name="Calculate Derived Features",
            description="Creates new features like moving averages, ratios, and growth rates",
            tags=["enrichment", "features", "calculations"]
        ),
        AgentSkill(
            id="merge_reference_data",
            name="Merge Reference Data",
            description="Merges dataset with reference data on common keys",
            tags=["enrichment", "merging", "reference-data"]
        )
    ]
)

DATA_ANALYST_AGENT_CARD = AgentCard(
    name="data_analyst_agent",
    description="Agent for performing statistical analysis and trend detection on datasets",
    url=f"http://localhost:{settings.data_analyst_port}",
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="analyze_dataset",
            name="Analyze Dataset",
            description="Performs comprehensive statistical analysis on a dataset",
            tags=["analysis", "statistics", "insights"]
        ),
        AgentSkill(
            id="detect_trends",
            name="Detect Trends",
            description="Identifies trends and patterns in time series data",
            tags=["analysis", "trends", "time-series"]
        ),
        AgentSkill(
            id="find_outliers",
            name="Find Outliers",
            description="Identifies outliers and anomalies in data using statistical methods",
            tags=["analysis", "outliers", "anomalies"]
        ),
        AgentSkill(
            id="calculate_correlations",
            name="Calculate Correlations",
            description="Analyzes correlations and relationships between variables",
            tags=["analysis", "correlation", "relationships"]
        ),
        AgentSkill(
            id="generate_summary_stats",
            name="Generate Summary Statistics",
            description="Calculates descriptive statistics and data summaries",
            tags=["analysis", "statistics", "summary"]
        )
    ]
)

PRESENTATION_AGENT_CARD = AgentCard(
    name="presentation_agent",
    description="Agent for creating reports, visualizations, and presentations from analysis results",
    url=f"http://localhost:10010",  # New port for presentation agent
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="create_report",
            name="Create Report",
            description="Creates a comprehensive report with visualizations and narrative",
            tags=["presentation", "reports", "visualization"]
        ),
        AgentSkill(
            id="generate_charts",
            name="Generate Charts",
            description="Creates various types of charts and visualizations",
            tags=["presentation", "charts", "visualization"]
        ),
        AgentSkill(
            id="write_narrative",
            name="Write Narrative",
            description="Generates human-readable narrative explaining analysis results",
            tags=["presentation", "narrative", "insights"]
        ),
        AgentSkill(
            id="compile_dashboard",
            name="Compile Dashboard",
            description="Assembles interactive dashboard with multiple visualizations",
            tags=["presentation", "dashboard", "interactive"]
        )
    ]
)

ORCHESTRATOR_AGENT_CARD = AgentCard(
    name="orchestrator_agent",
    description="Orchestrator agent that coordinates data analysis workflows via A2A protocol",
    url=f"http://localhost:{settings.orchestrator_port}",
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="coordinate_full_pipeline",
            name="Coordinate Full Pipeline",
            description="Orchestrates the complete data analysis pipeline from loading to presentation",
            tags=["orchestration", "pipeline", "coordination"]
        ),
        AgentSkill(
            id="route_task_to_agent",
            name="Route Task to Agent",
            description="Routes specific tasks to appropriate specialized agents via A2A",
            tags=["orchestration", "routing", "a2a"]
        ),
        AgentSkill(
            id="manage_data_handles",
            name="Manage Data Handles",
            description="Manages data handle lifecycle and inter-agent data sharing",
            tags=["orchestration", "data-handles", "management"]
        )
    ]
)

# Registry of all agent cards
ALL_AGENT_CARDS = [
    DATA_LOADER_AGENT_CARD,
    DATA_CLEANING_AGENT_CARD,
    DATA_ENRICHMENT_AGENT_CARD,
    DATA_ANALYST_AGENT_CARD,
    PRESENTATION_AGENT_CARD,
    ORCHESTRATOR_AGENT_CARD
] 