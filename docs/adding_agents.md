# Adding New Agents to the A2A/MCP System

This guide provides step-by-step instructions for adding new agents to the A2A/MCP data analysis system.

## 📋 **Prerequisites**

- Understanding of the A2A (Agent-to-Agent) protocol for inter-agent communication
- Understanding of MCP (Model-Context-Protocol) for external tool access
- Python 3.9+ development environment
- Basic knowledge of FastAPI and asyncio

## 🏗️ **Agent Architecture Overview**

Each agent in the system follows this standard structure:

```
new-agent/
├── pyproject.toml              # Dependencies and build configuration
├── README.md                   # Agent-specific documentation
├── new_agent/                  # Main agent package
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # A2A server entry point
│   ├── agent.py               # Core agent logic (internal algorithms)
│   └── prompt.py              # Agent prompts and descriptions
└── tests/                     # Unit tests (optional)
    └── test_tools.py
```

## 🚀 **Step-by-Step Agent Creation**

### **Step 1: Create Agent Directory Structure**

   ```bash
# From the project root
mkdir new-agent
cd new-agent
mkdir new_agent
mkdir tests
```

### **Step 2: Create pyproject.toml**

Create `pyproject.toml` with the standard dependencies:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "new-agent"
version = "0.1.0"
dependencies = [
    # Core A2A dependencies
    "a2a-sdk>=0.2.12",
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "httpx>=0.28.0",
    "pydantic>=2.10.0",
    "python-dotenv>=1.0.1",
    
    # Add your specific dependencies here
    # "pandas",
    # "numpy",
    # "requests",
]

[project.scripts]
start = "new_agent.__main__:main"
```

### **Step 3: Create Package __init__.py**

Create `new_agent/__init__.py`:

```python
"""New Agent Package"""

from .agent import NewAgent

__version__ = "1.0.0"
__all__ = ["NewAgent"]
```

### **Step 4: Create Core Agent Logic**

Create `new_agent/agent.py` with your business logic:

```python
"""New Agent Implementation - Internal Logic with MCP for External Tools"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)

class NewAgent:
    """
    New Agent that handles [specific domain] tasks using internal logic.
    External services are accessed via MCP clients.
    """
    
    def __init__(self):
        self.name = "new_agent"
        self.description = "Agent that performs [specific domain] operations"
        
        # MCP client for external services (if needed)
        self.mcp_client = None
        self._initialize_mcp_client()
        
    def _initialize_mcp_client(self):
        """Initialize MCP client for external services access."""
        try:
            import os
            mcp_url = os.getenv("NEW_AGENT_MCP_URL", "http://localhost:10011")
            mcp_api_key = os.getenv("MCP_API_KEY")
            if mcp_api_key:
                self.mcp_client = httpx.AsyncClient(
                    base_url=mcp_url,
                    headers={"X-API-Key": mcp_api_key}
                )
                logger.info("MCP client initialized for external services access")
        except Exception as e:
            logger.warning(f"MCP client initialization failed: {e}")
    
    async def your_main_skill(self, task_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main skill implementation for your agent.
        
        Args:
            task_params: Dictionary containing:
                - data_handle_id: ID of the input data handle
                - operation_params: Parameters for your operation
                - output_path: Optional path to save results
                
        Returns:
            Dictionary with operation results and new data handle
        """
        try:
            logger.info("Starting [your operation] process")
            
            # Extract parameters
            data_handle_id = task_params.get("data_handle_id")
            operation_params = task_params.get("operation_params", {})
            output_path = task_params.get("output_path")
            
            if not data_handle_id:
                raise ValueError("data_handle_id is required")
            
            # Load data using data handle manager (internal)
            from common_utils.data_handle_manager import get_data_handle_manager
            data_manager = get_data_handle_manager()
            
            data_handle = await data_manager.get_handle(data_handle_id)
            if not data_handle:
                raise ValueError(f"Data handle {data_handle_id} not found")
            
            # Your internal processing logic here
            # Example: Load and process data
            # result_data = your_processing_function(data_handle.local_path)
            
            # Example: Save results and create new data handle
            # Save processed data
            import tempfile
            import os
            if output_path:
                save_path = output_path
            else:
                save_path = os.path.join(tempfile.gettempdir(), f"processed_{data_handle_id}.csv")
            
            # your_save_function(result_data, save_path)
            
            # Create new data handle for processed data
            processed_handle = await data_manager.create_handle(
                data_type="processed_dataset",
                local_path=save_path,
                metadata={
                    "source_data_handle": data_handle_id,
                    "operation": "your_operation",
                    "parameters": operation_params
                }
            )
            
            result = {
                "status": "completed",
                "processed_data_handle_id": processed_handle.handle_id,
                "operation_performed": "your_operation"
            }
            
            logger.info("Operation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during operation: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _call_external_service_via_mcp(self, service_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Example of calling external service via MCP (if needed)."""
        if not self.mcp_client:
            logger.warning("MCP client not available")
            return None
        
        try:
            response = await self.mcp_client.post("/tools/external_service", json=service_params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"External service error: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling external service via MCP: {e}")
            return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.mcp_client:
            await self.mcp_client.aclose()
```

### **Step 5: Create A2A Server Entry Point**

Create `new_agent/__main__.py`:

```python
#!/usr/bin/env python3
"""
New Agent - A2A Server Implementation

This module provides the main entry point for the new agent using A2A protocol.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory for common_utils access
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import A2A SDK
from a2a import A2AServer
from a2a.types import AgentCard, AgentSkill

# Import local agent implementation
from .agent import NewAgent
from common_utils.agent_cards import NEW_AGENT_CARD  # You'll need to add this
from common_utils.types import DataHandle
from common_utils.data_handle_manager import get_data_handle_manager

logger = logging.getLogger(__name__)

class NewAgentA2AServer:
    """A2A Server wrapper for the New Agent."""
    
    def __init__(self):
        self.agent = NewAgent()
        self.server = None
        
    async def start_server(self, host: str = "localhost", port: int = 10011):
        """Start the A2A server."""
        
        # Define agent skills
        skills = [
            AgentSkill(
                id="your_main_skill",
                name="Your Main Skill",
                description="Description of what your main skill does",
                tags=["data_processing", "your_domain"]
            )
        ]
        
        # Create agent card
        agent_card = AgentCard(
            name="new_agent",
            description="Agent that performs [specific domain] operations",
            url=f"http://{host}:{port}",
            version="1.0.0",
            skills=skills
        )
        
        # Initialize A2A server
        self.server = A2AServer(agent_card)
        
        # Register skill handlers
        self.server.register_skill("your_main_skill", self._handle_main_skill)
        
        # Start server
        await self.server.start(host=host, port=port)
        logger.info(f"New Agent A2A server started on {host}:{port}")
    
    async def _handle_main_skill(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the main skill request."""
        try:
            logger.info(f"Executing main skill with params: {request}")
            
            # Extract task parameters
            task_params = request.get("parameters", {})
            
            # Execute the skill
            result = await self.agent.your_main_skill(task_params)
            
            return {
                "status": "completed" if result.get("status") == "completed" else "failed",
                "results": result,
                "agent_name": "new_agent"
            }
            
        except Exception as e:
            logger.error(f"Error executing main skill: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "agent_name": "new_agent"
            }

async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get port from environment or use default
    import os
    port = int(os.getenv("NEW_AGENT_PORT", "10011"))
    
    # Create and start server
    server = NewAgentA2AServer()
    
    try:
        await server.start_server(port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### **Step 6: Create Agent Prompts**

Create `new_agent/prompt.py`:

```python
"""Prompts and descriptions for the New Agent."""

NEW_AGENT_PROMPT = """
You are a specialized agent responsible for [specific domain] operations.

Your capabilities include:
- [Capability 1]
- [Capability 2] 
- [Capability 3]

You should:
1. Process data efficiently using internal algorithms
2. Use MCP clients only for external service access
3. Return structured results with proper data handles
4. Log all operations for audit trails

Always ensure data integrity and provide clear error messages if operations fail.
"""

NEW_AGENT_DESCRIPTION = "Specialized agent for [specific domain] data processing and analysis"
```

### **Step 7: Add Agent to System Registry**

Add your agent to `common_utils/agent_cards.py`:

```python
# Add to the imports section
NEW_AGENT_CARD = AgentCard(
    name="new_agent",
    description="Agent for [specific domain] operations",
    url=f"http://localhost:{settings.new_agent_port}",
    version="1.0.0",
    capabilities=AgentCapabilities(),
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="your_main_skill",
            name="Your Main Skill",
            description="Performs [specific operation] on datasets",
            tags=["data_processing", "your_domain"]
        )
    ]
)

# Add to ALL_AGENT_CARDS list
ALL_AGENT_CARDS = [
    DATA_LOADER_AGENT_CARD,
    DATA_CLEANING_AGENT_CARD,
    DATA_ENRICHMENT_AGENT_CARD,
    DATA_ANALYST_AGENT_CARD,
    PRESENTATION_AGENT_CARD,
    NEW_AGENT_CARD  # Add your agent here
]
```

### **Step 8: Update System Configuration**

Add your agent port to `common_utils/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    new_agent_port: int = Field(default=10011, description="New Agent port")
```

### **Step 9: Update Orchestrator (Optional)**

If your agent should be part of automated workflows, update `orchestrator_agent/scheduler.py` to include workflows that use your agent.

### **Step 10: Create Tests**

Create `tests/test_tools.py`:

```python
"""Tests for New Agent functionality."""

import pytest
import asyncio
from new_agent.agent import NewAgent

@pytest.mark.asyncio
async def test_main_skill():
    """Test the main skill functionality."""
    agent = NewAgent()
    
    # Mock test data
    test_params = {
        "data_handle_id": "test_handle",
        "operation_params": {"test": "value"}
    }
    
    # Test would require actual data handle setup
    # This is a placeholder for the test structure
    assert True  # Replace with actual test logic

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in agent."""
    agent = NewAgent()
    
    # Test with invalid parameters
    result = await agent.your_main_skill({})
    assert result["status"] == "failed"
    assert "error" in result
```

### **Step 11: Create README**

Create `README.md`:

```markdown
# New Agent

Description of what your agent does and its purpose in the data analysis pipeline.

## Features

- Feature 1
- Feature 2
- Feature 3

## Configuration

Set the following environment variables:

- `NEW_AGENT_PORT`: Port for the A2A server (default: 10011)
- `NEW_AGENT_MCP_URL`: URL for external MCP services (if needed)
- `MCP_API_KEY`: API key for MCP authentication

## Usage

Start the agent:

```bash
cd new-agent
python -m new_agent
```

Or using the script:

   ```bash
cd new-agent
start
```

## API

The agent exposes the following A2A skills:

### your_main_skill

Performs [operation description].

**Parameters:**
- `data_handle_id` (required): ID of input data
- `operation_params` (optional): Operation-specific parameters

**Returns:**
- `status`: "completed" or "failed"
- `processed_data_handle_id`: ID of processed data
- `operation_performed`: Name of operation
```

## 🔧 **Agent Integration Checklist**

When adding a new agent, ensure you:

- [ ] **Structure**: Follow the standard directory structure
- [ ] **Dependencies**: Include all required A2A dependencies in pyproject.toml
- [ ] **A2A Server**: Implement proper A2A server with skill registration
- [ ] **Internal Logic**: Keep data processing logic internal to the agent
- [ ] **MCP Integration**: Use MCP only for external services (databases, APIs)
- [ ] **Data Handles**: Use the data handle system for data flow
- [ ] **Logging**: Implement proper logging for observability
- [ ] **Error Handling**: Provide comprehensive error handling
- [ ] **Documentation**: Create clear README and code documentation
- [ ] **Registry**: Add agent card to system registry
- [ ] **Testing**: Implement unit tests for agent functionality
- [ ] **Security**: Follow security best practices (input validation, etc.)
- [ ] **Monitoring**: Ensure agent is compatible with observability system

## 🚀 **Scaling Considerations**

### **Horizontal Scaling**

- Each agent can be deployed independently
- Use environment variables for port configuration
- Implement health checks for load balancer integration
- Consider using container orchestration (Docker/Kubernetes)

### **Performance Optimization**

- Use async/await for all I/O operations
- Implement connection pooling for external services
- Cache frequently used data when appropriate
- Monitor and optimize memory usage

### **High Availability**

- Implement circuit breakers for external dependencies
- Add retry logic with exponential backoff
- Use proper timeout configurations
- Implement graceful shutdown handling

### **Security**

- Validate all input parameters
- Use secure communication (HTTPS/TLS)
- Implement proper authentication and authorization
- Regular security audits and updates

## 📊 **Monitoring and Observability**

Your agent will automatically integrate with the system's observability stack:

- **Distributed Tracing**: OpenTelemetry integration for request tracing
- **Metrics**: Prometheus metrics for performance monitoring  
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Built-in health endpoints for monitoring

## 🤝 **Contributing**

When contributing new agents:

1. Follow the established patterns and conventions
2. Write comprehensive tests
3. Update documentation
4. Ensure security best practices
5. Test integration with existing agents
6. Update system configuration as needed

This standardized approach ensures all agents in the system are consistent, maintainable, and scalable. 