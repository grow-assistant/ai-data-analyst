# 🛠️ Utility Scripts Directory

This directory contains utility scripts and tools for the Multi-Agent Data Analysis Framework.

## 📋 Available Scripts

### 🚀 Dashboard & UI Scripts
- **`launch_dashboard.py`**: Robust dashboard launcher with path and dependency management
  - Automatically handles Python path configuration
  - Checks and installs missing dependencies
  - Usage: `python scripts/launch_dashboard.py`

- **`run_dashboard.py`**: Original Streamlit dashboard launcher
  - Provides web interface for interactive data analysis
  - Auto-installs required packages if missing
  - Usage: `python scripts/run_dashboard.py`
  
- **`streamlit_dashboard.py`**: Main Streamlit dashboard application
  - Interactive web interface for data analysis with natural language queries
  - Dataset selection, analysis configuration, and results visualization
  - Usage: `streamlit run scripts/streamlit_dashboard.py`

### 🔧 CLI & Framework Tools
- **`framework_cli.py`**: Main command-line interface for the framework
  - Complete pipeline operations and dataset processing
  - Health checks and agent status monitoring
  - Usage: `python scripts/framework_cli.py data/Superstore.csv`

### 🧪 Testing & Validation Scripts
- **`test_schema_profiler_ai.py`**: Test script for AI-powered schema profiler
  - Demonstrates intelligent dataset profiling capabilities
  - Tests configuration caching and reuse functionality
  - Usage: `python scripts/test_schema_profiler_ai.py`

## 🔧 Usage Instructions

### Prerequisites
1. Ensure all agents are running: `powershell ./start_all_agents.ps1`
2. Set environment variables (especially `GOOGLE_API_KEY` for AI features)

### Running Scripts
```bash
# From the agents root directory:

# Launch Streamlit dashboard
python scripts/run_dashboard.py

# Run dashboard directly
streamlit run scripts/streamlit_dashboard.py

# Use CLI interface
python scripts/framework_cli.py --check

# Test AI schema profiler
python scripts/test_schema_profiler_ai.py
```

### Development Scripts
These scripts are helpful for development and testing:
- Test individual agent functionality
- Validate multi-agent communication
- Demonstrate framework capabilities
- Provide examples for integration

## 📝 Adding New Scripts

When adding new utility scripts:
1. Place them in this `scripts/` directory
2. Add documentation to this README
3. Include usage examples
4. Follow the naming convention: `verb_noun.py` (e.g., `test_agent.py`, `run_analysis.py`)
5. Add proper error handling and help messages

## 🔗 Related Documentation
- [Main README](../README.md) - Framework overview
- [Dashboard README](../docs/STREAMLIT_DASHBOARD_README.md) - Dashboard usage guide
- [CLI Usage Guide](../docs/CLI_USAGE_GUIDE.md) - Command-line interface
- [Project Structure](../docs/PROJECT_STRUCTURE.md) - Complete codebase organization

---
*Part of the Multi-Agent Data Analysis Framework* 