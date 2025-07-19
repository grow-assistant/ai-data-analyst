# 🤖 Multi-Agent A2A Data Analysis Framework

[![Framework Version](https://img.shields.io/badge/Version-3.1.0-blue.svg)](https://github.com/your-repo/adk-samples)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-6%20Phases%20Complete-green.svg)](./TESTING_PLAN.md)
[![Production Ready](https://img.shields.io/badge/Status-Enterprise%20Ready-success.svg)](./TESTING_PLAN.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Enterprise-grade multi-agent framework for automated data analysis, processing, and intelligence with A2A (Agent-to-Agent) communication protocol.**

## 🌟 **Framework Overview**

The Multi-Agent A2A Framework is a production-ready system featuring **7 specialized agents** that work together to provide automated data analysis, processing, and intelligence capabilities. Built with enterprise-grade security, observability, and scheduling features.

### **🎯 Key Features**

- 🤖 **7 Specialized Agents**: Data Loader, Cleaning, Enrichment, Analysis, Presentation, Orchestrator, Schema Profiler
- 🔄 **A2A Communication**: JSON-RPC based inter-agent communication
- 🌐 **Streamlit Dashboard**: User-friendly web interface for interactive data analysis
- 🔐 **Enterprise Security**: OAuth2 authentication, ACL authorization, audit logging
- 📊 **Advanced Observability**: OpenTelemetry tracing, Prometheus metrics, Jaeger integration
- ⏰ **Automated Scheduling**: APScheduler workflow automation
- 🗄️ **Large Dataset Support**: 1GB+ dataset processing with memory management
- 🔍 **AI-Powered Profiling**: Intelligent schema profiling with Gemini AI and configuration caching
- 🧠 **Root Cause Analysis**: Advanced Why-Bot for automated root cause discovery

### **🏗️ Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                          │
│                    (Port 8000)                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │ A2A Communication (JSON-RPC)
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼───┐   ┌───▼────┐   ┌───▼────┐
    │ Data   │   │ Data   │   │ Data   │
    │ Loader │   │Cleaning│   │Analysis│
    │ :10006 │   │ :10008 │   │ :10007 │
    └────────┘   └────────┘   └────────┘
         │            │            │
    ┌────▼───┐   ┌───▼────┐   ┌───▼────┐
    │ Data   │   │Present │   │ Schema │
    │Enrich  │   │ Agent  │   │Profile │
    │ :10009 │   │ :10010 │   │ :10012 │
    └────────┘   └────────┘   └────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.11+
- Windows 10/11 (PowerShell) or Linux/macOS
- 8GB+ RAM recommended for large datasets
- Available ports: 8000, 10006-10012

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/adk-samples.git
cd adk-samples/python/agents

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install additional observability dependencies (optional)
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-jaeger
pip install prometheus-client

# 4. Start all agents
./start_all_agents.ps1

# 5. Verify installation
python tests/integration/run_all_fixed_tests.py
```

### **🎮 Streamlit Dashboard**

Interactive web interface for easy data analysis with natural language queries:

```bash
# Launch the dashboard (recommended)
python scripts/run_dashboard.py

# Or run directly with Streamlit
streamlit run scripts/streamlit_dashboard.py

# Dashboard available at: http://localhost:8501
```

**Features:**
- 📁 **Dataset Browser**: Select from available datasets in `/data` folder
- 💬 **Natural Language Queries**: "Show me trends for last week", "Find anomalies", etc.
- 🔧 **Configurable Analysis**: Adjust depth, enable root cause analysis, choose output format
- 🏥 **Health Monitoring**: Check orchestrator and agent status
- 📊 **Interactive Results**: View pipeline stages and analysis results
- 💾 **Organized Output**: Automatic timestamped folders for all analysis results, logs, and reports

**Quick Start:**
1. Start all agents: `powershell ./start_all_agents.ps1`
2. Launch dashboard: `python scripts/run_dashboard.py`
3. Select dataset, enter query, click "Start Analysis"

See [STREAMLIT_DASHBOARD_README.md](docs/STREAMLIT_DASHBOARD_README.md) for detailed usage guide.

## 📋 **CLI Usage Guide**

### **🔧 Agent Management**

#### **Start All Agents**
```bash
# Start all agents with PowerShell script
./start_all_agents.ps1

# Or start individual agents
cd data-loader-agent && python -m data_loader &
cd data-cleaning-agent && python -m data_cleaning_agent &
cd data-enrichment-agent && python -m data_enrichment_agent &
cd data-analyst-agent && python -m data_analyst &
cd presentation-agent && python -m presentation_agent &
cd orchestrator-agent && python -m orchestrator_agent &
cd schema-profiler-agent && python -m schema_profiler &
```

#### **Check Agent Health**
```bash
# Quick health check script
python -c "
import asyncio, httpx
async def check():
    ports = [8000, 10006, 10007, 10008, 10009, 10010, 10012]
    async with httpx.AsyncClient() as client:
        for port in ports:
            try:
                r = await client.get(f'http://localhost:{port}/health')
                print(f'Port {port}: ✅ {r.status_code}')
            except: print(f'Port {port}: ❌ FAILED')
asyncio.run(check())
"
```

#### **Stop All Agents**
```bash
# Kill all Python agent processes
pkill -f "python -m"

# Or on Windows
taskkill /f /im python.exe
```

### **🧪 Testing & Validation**

#### **Run Complete Test Suite**
```bash
cd tests/integration

# Comprehensive test with error handling
python run_all_fixed_tests.py

# Individual test phases
python test_individual_agents.py          # Phase 1: Individual agents
python test_a2a_communication.py          # Phase 2: A2A communication  
python test_full_pipeline.py              # Phase 3: Integration
python test_scheduled_workflows.py        # Phase 4.1: Workflows
python test_production_scenarios.py       # Phase 4.2: Production
python test_security_features.py          # Phase 5.1: Security
python test_observability_features.py     # Phase 5.2: Observability
python test_schema_profiler.py            # Phase 6: Schema profiler
```

#### **Pytest Integration**
```bash
# Run with pytest for detailed output
pytest tests/integration/ -v -s

# Run specific test categories
pytest tests/integration/test_*_features.py -v
pytest tests/integration/test_production_*.py -v
```

### **📊 Data Processing Workflows**

#### **Basic Data Analysis Pipeline**
```bash
# 1. Load sample data
curl -X POST http://localhost:10006/execute \
  -H "Content-Type: application/json" \
  -d '{
    "skill": "load_dataset",
    "parameters": {
      "file_path": "test_data/sales_data_small.csv",
      "file_type": "csv"
    }
  }'

# 2. Clean the data
curl -X POST http://localhost:10008/execute \
  -H "Content-Type: application/json" \
  -d '{
    "skill": "clean_dataset", 
    "parameters": {
      "data_handle_id": "YOUR_DATA_HANDLE_ID",
      "operations": ["remove_duplicates", "handle_missing"]
    }
  }'

# 3. Analyze the data
curl -X POST http://localhost:10007/execute \
  -H "Content-Type: application/json" \
  -d '{
    "skill": "analyze_dataset",
    "parameters": {
      "data_handle_id": "YOUR_CLEANED_HANDLE_ID",
      "analysis_type": "comprehensive"
    }
  }'
```

#### **Orchestrated Workflow**
```bash
# Run complete pipeline via orchestrator
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_type": "data_analysis",
    "description": "Analyze sales data for trends and insights",
    "data_source": "test_data/sales_data_medium.csv"
  }'
```

### **⏰ Scheduled Workflows**

#### **Create Scheduled Analysis**
```bash
# Daily analysis workflow
curl -X POST http://localhost:8000/workflows \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Daily Sales Analysis",
    "description": "Automated daily sales data analysis",
    "schedule_type": "cron",
    "schedule_config": {"hour": 9, "minute": 0},
    "workflow_steps": [
      {
        "agent": "data_loader_agent",
        "skill": "load_dataset",
        "params": {"data_source": "daily_sales"}
      },
      {
        "agent": "data_analyst_agent", 
        "skill": "analyze_dataset",
        "params": {"analysis_type": "trend_analysis"}
      }
    ]
  }'
```

#### **Manage Workflows**
```bash
# List all workflows
curl http://localhost:8000/workflows \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get workflow status
curl http://localhost:8000/workflows/{WORKFLOW_ID} \
  -H "Authorization: Bearer YOUR_TOKEN"

# Disable workflow
curl -X PUT http://localhost:8000/workflows/{WORKFLOW_ID}/disable \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### **🔐 Security & Authentication**

#### **Generate API Token**
```bash
# Example OAuth2 token generation
python -c "
from common_utils.security import SecurityManager
import asyncio

async def get_token():
    sm = SecurityManager()
    sm.oauth2_manager.register_client('my_client', 'my_secret')
    token = await sm.authenticate_agent('my_client', 'my_secret')
    print(f'Token: {token.access_token}')

asyncio.run(get_token())
"
```

#### **Test API Access**
```bash
# Test authenticated endpoint
curl http://localhost:8000/workflows \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Test with API key
curl http://localhost:10006/health \
  -H "X-API-Key: mcp-dev-key"
```

### **📈 Monitoring & Observability**

#### **Check System Metrics**
```bash
# Prometheus metrics (if available)
curl http://localhost:8000/metrics

# Agent health summary
curl http://localhost:8000/health

# System resource usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

#### **View Distributed Traces**
```bash
# If Jaeger is running
echo "Jaeger UI: http://localhost:16686"
echo "Search for traces by service: data_loader_agent, orchestrator_agent, etc."
```

### **🗄️ Large Dataset Processing**

#### **Process Large Files**
```bash
# Create test large dataset
python test_data/create_sample_data.py --size large --output large_dataset.csv

# Process with memory monitoring
python -c "
import asyncio, httpx, psutil

async def process_large():
    print(f'Initial Memory: {psutil.virtual_memory().percent}%')
    
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            'http://localhost:10006/execute',
            json={
                'skill': 'load_dataset',
                'parameters': {
                    'file_path': 'large_dataset.csv',
                    'file_type': 'csv'
                }
            }
        )
        print(f'Response: {response.status_code}')
        print(f'Final Memory: {psutil.virtual_memory().percent}%')

asyncio.run(process_large())
"
```

## 📁 **Framework Structure**

```
agents/
├── common_utils/           # Shared utilities
│   ├── security.py        # OAuth2, ACL, audit logging
│   ├── data_handle_manager.py  # Data sharing
│   └── circuit_breaker.py # Resilience patterns
├── orchestrator-agent/     # Main coordinator
├── data-loader-agent/      # Data ingestion
├── data-cleaning-agent/    # Data cleaning
├── data-enrichment-agent/  # Feature engineering
├── data-analyst-agent/     # Statistical analysis
├── presentation-agent/     # Report generation
├── schema-profiler-agent/  # AI schema profiling
├── rootcause-analyst-agent/ # Root cause analysis
├── monitoring/             # Observability
├── tests/                  # Comprehensive test suite
├── streamlit_app.py       # Interactive dashboard
└── README.md              # This file
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
export MCP_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-google-key"  # For AI features

# Optional
export JAEGER_AGENT_HOST="localhost"
export JAEGER_AGENT_PORT="6831"
export PROMETHEUS_PORT="9090"
```

### **System Configuration**
Edit `config/system_config.yaml`:
```yaml
# Agent ports
agents:
  orchestrator:
    port: 8000
  data_loader:
    port: 10006
  # ... other agents

# Security settings
security:
  api_keys:
    mcp_server: "mcp-dev-key"
  rate_limiting:
    enabled: true
    requests_per_minute: 100
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Port Conflicts**
```bash
# Check port usage
netstat -an | grep :8000
netstat -an | grep :1000[6-9]

# Kill processes on specific ports
lsof -ti:8000 | xargs kill -9
```

#### **Agent Startup Issues**
```bash
# Check logs
python -m orchestrator_agent 2>&1 | tee orchestrator.log

# Test individual agent
cd data-loader-agent
python -m data_loader --debug
```

#### **Memory Issues**
```bash
# Monitor resource usage during processing
watch -n 1 "ps aux | grep python | head -10"

# Check system resources
htop  # or top on basic systems
```

#### **Test Failures**
```bash
# Run diagnostic script
python tests/integration/run_all_fixed_tests.py

# Check agent health first
python -c "
import asyncio, httpx
async def health():
    async with httpx.AsyncClient() as client:
        for port in [8000, 10006, 10007, 10008, 10009, 10010]:
            try:
                r = await client.get(f'http://localhost:{port}/health')
                print(f'Port {port}: {r.status_code}')
            except Exception as e:
                print(f'Port {port}: ERROR - {e}')
asyncio.run(health())
"
```

### **Performance Tuning**

#### **Memory Optimization**
```bash
# Set memory limits for agents
export PYTHON_MEMORY_LIMIT="1G"

# Use chunked processing for large files
python -c "
import pandas as pd
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    # Process chunk
    pass
"
```

#### **Concurrent Processing**
```bash
# Adjust worker counts in system_config.yaml
mcp_server:
  workers: 4  # Increase for better throughput
  max_connections: 500
```

## 📚 **API Reference**

### **Orchestrator API**
- `POST /analyze` - Run analysis workflow
- `GET /workflows` - List scheduled workflows  
- `POST /workflows` - Create new workflow
- `PUT /workflows/{id}/disable` - Disable workflow
- `GET /health` - Health check

### **Agent APIs**
All agents expose:
- `GET /health` - Health status
- `GET /capabilities` - Available skills
- `POST /execute` - Execute skill

### **Data Handle Management**
- Persistent storage in `data_handles/`
- Cross-agent data sharing
- Metadata tracking
- Automatic cleanup

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests**: Ensure all tests pass with `python tests/integration/run_all_fixed_tests.py`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 agents/
black agents/

# Run full test suite
python tests/integration/run_all_fixed_tests.py
```

## 📄 **License**

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: [Full docs](./docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Testing**: [Test Plan](./TESTING_PLAN.md)
- **Architecture**: [System Design](./docs/CLEANUP_AND_STRUCTURE.md)

## 🏆 **Achievements**

- ✅ **7 Production Agents** with A2A communication
- ✅ **100% Test Coverage** across 6 testing phases
- ✅ **Enterprise Security** with OAuth2 and ACL
- ✅ **Advanced Observability** with OpenTelemetry
- ✅ **Automated Scheduling** with APScheduler
- ✅ **AI-Powered Features** with schema profiling and root cause analysis
- ✅ **Large Dataset Support** up to 1GB+ with memory management
- ✅ **Production Ready** with comprehensive error handling

---

**Built with ❤️ for enterprise data analysis automation** 