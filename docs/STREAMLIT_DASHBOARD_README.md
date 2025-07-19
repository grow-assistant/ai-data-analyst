# 🤖 Multi-Agent Data Analysis Dashboard

A Streamlit web interface for the Multi-Agent Data Analysis Framework that allows you to interact with the orchestrator agent to analyze datasets through a user-friendly web interface.

## ✨ Features

- **📁 Dataset Selection**: Browse and select from available datasets in the `/data` folder
- **💬 Natural Language Queries**: Describe your analysis needs in plain English
- **🔍 Multi-Agent Pipeline**: Automatically orchestrates data loading, cleaning, enrichment, analysis, and presentation
- **🏥 Health Monitoring**: Check the status of the orchestrator and agent system
- **📊 Interactive Results**: View analysis results with detailed pipeline information
- **🔧 Configurable Options**: Adjust analysis depth, output format, and enable root cause analysis

## 🚀 Quick Start

### Prerequisites

1. **All agents must be running**:
   ```powershell
   powershell ./start_all_agents.ps1
   ```

2. **Required Python packages** (auto-installed):
   - streamlit
   - httpx  
   - pandas

### Running the Dashboard

**Option 1: Use the launcher script (recommended)**
```bash
python run_dashboard.py
```

**Option 2: Run Streamlit directly**
```bash
streamlit run scripts/streamlit_dashboard.py
```

The dashboard will be available at: http://localhost:8501

## 📖 How to Use

### 1. Dataset Selection
- The dashboard automatically scans the `/data` folder for supported file types:
  - `.csv` files
  - `.json` files  
  - `.tdsx` files (Tableau Data Source)
- Select your desired dataset from the dropdown
- View file information (size, modification date, type)

### 2. Analysis Configuration
Use the sidebar to configure your analysis:

- **Include Root Cause Analysis**: Enable the Why-Bot for automated root cause discovery
- **Analysis Depth**: Choose between `quick`, `standard`, or `comprehensive`
- **Output Format**: Select `html`, `pdf`, or `markdown` for the final report

### 3. Analysis Request
- Enter your analysis query in natural language
- Use the example queries for inspiration
- Examples:
  - "Show me the trends for the last month"
  - "Find anomalies and their root causes"
  - "Analyze sales performance by region"
  - "What factors drive revenue changes?"

### 4. Execute Analysis
- Click "🔍 Start Analysis" to begin the multi-agent pipeline
- Monitor progress through the real-time status updates
- View detailed results when the analysis completes

## 🔄 Analysis Pipeline

The dashboard orchestrates a sophisticated multi-agent pipeline:

1. **📁 Data Loading**: Loads your dataset (with Tableau Hyper API support for .tdsx files)
2. **🧹 Data Cleaning**: Cleans and validates the data
3. **🔄 Data Enrichment**: Enriches data with additional features
4. **🔬 Data Analysis**: Performs comprehensive analysis with multiple modules
5. **🔍 Root Cause Analysis**: Uses Why-Bot for automated root cause discovery (optional)
6. **📊 Presentation**: Generates executive reports with Google Gemini AI

## 🏥 System Health

Use the "Check Health" button in the sidebar to:
- Verify orchestrator connectivity
- View agent capabilities
- Troubleshoot connection issues

## 🛠️ Troubleshooting

### Dashboard Won't Start
- Ensure you're in the correct directory (should contain `/data` folder)
- Check that Python and required packages are installed
- Run `python run_dashboard.py` for automatic dependency installation

### Orchestrator Not Responding
- Verify all agents are running: `powershell ./start_all_agents.ps1`
- Check that the orchestrator is running on port 10000
- Wait for all agents to be fully initialized (can take 1-2 minutes)

### Analysis Fails
- Ensure the selected dataset file exists and is accessible
- Check agent logs for detailed error information
- Verify sufficient system resources for large datasets

### Connection Timeout
- Large datasets may take several minutes to process
- The dashboard has a 5-minute timeout for analysis operations
- Monitor agent logs for progress updates

## 📂 Supported File Types

- **CSV**: Standard comma-separated values
- **JSON**: Structured JSON data files  
- **TDSX**: Tableau Data Source files (uses Hyper API for fast loading)

## 🔧 Configuration

The dashboard connects to:
- **Orchestrator Agent**: http://localhost:10000
- **Data Folder**: `./data` (relative to script location)

## 📊 Understanding Results

The dashboard displays:
- **Pipeline Metadata**: ID, timestamps, completion status
- **Stage Details**: Status and data handles for each pipeline stage
- **Analysis Summary**: Key findings and metrics
- **Final Reports**: Links to detailed HTML/PDF reports (when available)

## 🚀 Advanced Usage

### Custom Analysis Configurations
The dashboard passes your configuration to the orchestrator:
```python
{
    "analysis_config": {
        "query": "your_analysis_query",
        "include_root_cause": True,
        "depth": "comprehensive", 
        "output_format": "html"
    }
}
```

### Integration with Existing Workflows
The dashboard uses the same orchestrator API that powers the CLI and other interfaces, ensuring consistency across all access methods.

## 📈 Performance Tips

- **Large Files**: TDSX files use optimized Hyper API loading
- **Resource Usage**: Monitor system resources during analysis
- **Parallel Processing**: The framework uses multiple agents for optimal performance
- **Caching**: Analysis results are cached in session state for quick review

## 🤝 Contributing

The dashboard is part of the larger Multi-Agent Data Analysis Framework. To contribute:
1. Follow the existing code patterns
2. Test with various dataset types
3. Ensure compatibility with the orchestrator API
4. Update documentation for new features

---

**Happy Analyzing! 🎉** 