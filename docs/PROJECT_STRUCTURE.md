# 🤖 Multi-Agent Data Analysis Framework

This document outlines the complete directory structure of the cleaned-up multi-agent data analysis framework, ready for GitHub.

## 📁 Root Directory

```
.
├── .gitignore
├── docs/
│   ├── CLI_USAGE_GUIDE.md
├── README.md
│   ├── STREAMLIT_DASHBOARD_README.md
│   ├── PROJECT_STRUCTURE.md
├── config.env
│   ├── framework_cli.py
│   ├── output_manager.py
├── requirements-streamlit.txt
├── start_all_agents.ps1
├── scripts/
│   ├── streamlit_dashboard.py
├── common_utils/
├── config/
├── data/
├── docs/
├── monitoring/
├── outputs/
├── scripts/
├── test_data/
├── tests/
├── data-analyst-agent/
├── data-cleaning-agent/
├── data-enrichment-agent/
├── data-loader-agent/
├── orchestrator-agent/
├── presentation-agent/
├── rootcause-analyst-agent/
└── schema-profiler-agent/
```

### **Core Files**
- **`.gitignore`**: Excludes temporary files, large datasets, and sensitive information.
- **`docs/CLI_USAGE_GUIDE.md`**: Instructions for using the main command-line interface.
- **`README.md`**: Main project overview, architecture, and quick start guide.
- **`docs/STREAMLIT_DASHBOARD_README.md`**: Detailed guide for the Streamlit dashboard.
- **`docs/PROJECT_STRUCTURE.md`**: Complete codebase organization documentation.
- **`config.env`**: Environment configuration file for API keys and settings.
- **`scripts/framework_cli.py`**: Main command-line interface for the framework.
- **`common_utils/output_manager.py`**: Utility for managing timestamped output folders.
- **`requirements-streamlit.txt`**: Python dependencies for the Streamlit dashboard.
- **`start_all_agents.ps1`**: PowerShell script to start all agents.
- **`scripts/streamlit_dashboard.py`**: Main Streamlit dashboard application.

### **Core Directories**
- **`common_utils/`**: Shared utilities for all agents (config, security, data handles).
- **`config/`**: System-level configurations.
- **`data/`**: Large datasets for analysis (excluded by `.gitignore`).
- **`docs/`**: Framework documentation.
- **`monitoring/`**: Observability and monitoring tools.
- **`outputs/`**: Timestamped analysis results, logs, and reports (excluded by `.gitignore`).
- **`scripts/`**: Utility scripts for testing and launching the dashboard.
- **`test_data/`**: Small sample datasets for testing.
- **`tests/`**: Integration and unit tests for the framework.

## 🤖 Agent Directories

Each agent follows a standard structure:

```
agent-name/
├── agent_name/
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent.py
│   ├── agent_executor.py
│   └── prompt.py
├── pyproject.toml
└── README.md
```

### **Agent List**
- **`data-analyst-agent/`**: Performs comprehensive data analysis.
- **`data-cleaning-agent/`**: Cleans and validates datasets.
- **`data-enrichment-agent/`**: Enriches data with additional features.
- **`data-loader-agent/`**: Loads data from various sources (CSV, TDSX).
- **`orchestrator-agent/`**: Coordinates the multi-agent pipeline.
- **`presentation-agent/`**: Generates reports and visualizations.
- **`rootcause-analyst-agent/`**: Performs root cause analysis (Why-Bot).
- **`schema-profiler-agent/`**: Intelligently profiles datasets and caches configurations.

## 🧹 Cleanup Summary

### **Removed Files**
- `install_enhancements.py` (Replaced by `pyproject.toml` dependencies)
- `extract_report.py` (Functionality integrated into `output_manager.py`)
- `load_env.py` (Handled by agents internally)
- `process_tdsx.py` (Superseded by `data-loader-agent`)
- `test_agent_simple.py` (Replaced by comprehensive tests)
- `cli.py` (Consolidated into `framework_cli.py`)
- `0.0.20063` (Temporary version file)
- `report_67ac9cfd.html` (Generated report)
- `tdsx_processing_result_AI_DS.json` (Temporary result file)

### **Removed Directories**
- `.cursor/` (Editor-specific)
- `.pytest_cache/` (Test cache)
- `__pycache__` directories (Handled by `.gitignore`)
- `venv/` directories (Handled by `.gitignore`)

### **Organized Files**
- Moved `run_dashboard.py` and `test_schema_profiler_ai.py` into `scripts/`.
- Renamed `tdsx_cli.py` to `framework_cli.py`.
- Updated documentation to reflect new structure.

The repository is now clean, organized, and ready for version control with GitHub. 