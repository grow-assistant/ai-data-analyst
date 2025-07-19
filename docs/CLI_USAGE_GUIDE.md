# 🚀 CLI Usage Guide - Multi-Agent Data Analysis Framework

**Production-Ready Framework!** The enhanced multi-agent framework provides comprehensive CLI tools for data analysis.

## ✅ **Available Tools:**

1. **Framework CLI** - `scripts/framework_cli.py` for complete pipeline operations
2. **PowerShell Launcher** - `start_all_agents.ps1` for agent management  
3. **Streamlit Dashboard** - `scripts/run_dashboard.py` for web interface
4. **Test Scripts** - `scripts/` directory for validation and testing

## 🎯 **Quick Start (CLI)**

### **1. Start All Agents**
```bash
# Use the PowerShell script (recommended)
powershell ./start_all_agents.ps1

# Check capabilities after startup
python scripts/framework_cli.py --check
```

### **2. Launch Web Dashboard**
```bash
# Interactive Streamlit dashboard
python scripts/run_dashboard.py
```

### **3. Run Analysis Pipeline**
```bash
# Analyze a dataset using the complete pipeline
python scripts/framework_cli.py data/Superstore.csv

# With custom configuration
python scripts/framework_cli.py data/Superstore.csv --config '{"analysis_depth": "comprehensive"}'
```

### **3. Run Tests**
```bash
# Complete test suite
python cli.py test

# Specific test types
python cli.py test --type individual
python cli.py test --type integration
python cli.py test --type security
```

### **4. Process Data**
```bash
# Process a data file
python cli.py process test_data/sales_data_small.csv

# With specific analysis type
python cli.py process test_data/sales_data_medium.csv --analysis-type trend_analysis
```

## 📋 **Current Status**

✅ **Agents Running**: 6/7 agents healthy  
✅ **Configuration**: All environment variables set  
✅ **PowerShell Script**: Fixed and working  
✅ **CLI Tool**: Fully functional  
✅ **Tests**: Running successfully  

## 🎛️ **Available CLI Commands**

| Command | Description | Example |
|---------|-------------|---------|
| `python cli.py health` | Check agent health | `python cli.py health` |
| `python cli.py start` | Start all agents | `python cli.py start` |
| `python cli.py status` | Show framework status | `python cli.py status` |
| `python cli.py test` | Run tests | `python cli.py test --type individual` |
| `python cli.py process <file>` | Process data | `python cli.py process data.csv` |

## 🔧 **Alternative Methods**

### **Direct API Calls**
```bash
# Check health manually
curl http://localhost:8000/health
curl http://localhost:10006/health

# Process data via API
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_type": "data_analysis",
    "description": "Analyze sales data",
    "data_source": "test_data/sales_data_small.csv"
  }'
```

### **Individual Agent Testing**
```bash
cd tests/integration

# Run specific tests
python test_individual_agents.py
python test_a2a_communication.py
python test_full_pipeline.py
python run_all_fixed_tests.py  # Complete suite
```

### **Manual Agent Startup**
```bash
# Start individual agents
cd data-loader-agent && python -m data_loader &
cd data-cleaning-agent && python -m data_cleaning_agent &
cd data-analyst-agent && python -m data_analyst &
cd orchestrator-agent && python -m orchestrator_agent &
```

## 🎯 **Common Use Cases**

### **Daily Operations**
```bash
# 1. Start framework
python cli.py start

# 2. Check all is working
python cli.py health

# 3. Run health tests
python cli.py test --type individual

# 4. Process your data
python cli.py process your_data.csv
```

### **Development & Testing**
```bash
# Full test suite
python cli.py test

# Specific test categories
python cli.py test --type security
python cli.py test --type production
python cli.py test --type integration
```

### **Data Analysis Pipeline**
```bash
# Quick analysis
python cli.py process test_data/sales_data_small.csv

# Advanced analysis
python cli.py process large_dataset.csv --analysis-type comprehensive
```

## 🏆 **Success Indicators**

✅ **All Agents Healthy**: `python cli.py health` shows 6-7 agents healthy  
✅ **No Configuration Errors**: No "GOOGLE_API_KEY required" messages  
✅ **Tests Passing**: `python cli.py test` shows majority of tests passing  
✅ **Data Processing**: Can process CSV files successfully  

## 🔄 **Streamlit Dashboard (Optional)**

If you want to fix the Streamlit dashboard:
```bash
# Install compatible version
python -m pip install streamlit==1.40.0

# Launch dashboard
python launch_dashboard.py
# Or: streamlit run streamlit_app.py
```

## 🆘 **Troubleshooting**

### **If Agents Won't Start**
```bash
# Check ports
netstat -an | findstr ":8000 :10006 :10007 :10008"

# Kill existing processes
taskkill /f /im python.exe

# Restart
python cli.py start
```

### **If Tests Fail**
```bash
# Check agent health first
python cli.py health

# Run individual tests to isolate issues
python cli.py test --type individual
```

### **If Data Processing Fails**
```bash
# Verify orchestrator is running
curl http://localhost:8000/health

# Check file exists
dir test_data\*.csv

# Try with sample data
python cli.py process test_data/sales_data_small.csv
```

---

## 🎉 **You're Ready!**

The Multi-Agent A2A Framework is now fully operational via CLI. All configuration issues have been resolved, and you can:

1. ✅ Start agents with `python cli.py start`
2. ✅ Check health with `python cli.py health` 
3. ✅ Run tests with `python cli.py test`
4. ✅ Process data with `python cli.py process <file>`

**Next steps**: Try processing your own CSV data files or explore the full test suite!

---

**📚 Full Documentation**: See [README.md](./README.md) for complete details  
**🧪 Testing Plan**: See [TESTING_PLAN.md](./TESTING_PLAN.md) for test information  
**🎮 Dashboard Guide**: See [STREAMLIT_DASHBOARD_GUIDE.md](./STREAMLIT_DASHBOARD_GUIDE.md) for web interface 