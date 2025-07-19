# Common Utils

Common utilities package for the multi-agent system. Provides shared A2A (Agent-to-Agent) communication logic and other utilities.

## Features

- **A2A Client**: Discover and communicate with A2A agents
- **A2A Server**: Expose agents as A2A services
- **Agent Configuration**: Standardized agent card definitions
- **FastAPI Integration**: Built-in web server for agent endpoints

## Installation

```bash
# From the agents directory
pip install -e ./common_utils

# Or with ADK support
pip install -e "./common_utils[adk]"
```

## Usage

### A2A Client

```python
from common_utils.a2a import A2AClient

async with A2AClient() as client:
    # Discover available agents
    agents = await client.discover_agents()
    
    # Send message to specific agent
    response = await client.send_message("data_loader_agent", "Load sample.csv")
    print(response)
```

### A2A Server

```python
from common_utils.a2a import A2AServer, DATA_LOADER_AGENT_CARD

# Create server
server = A2AServer(agent=None, agent_card=DATA_LOADER_AGENT_CARD)

# Run server
await server.run(port=10006)
```

## Agent Cards

Pre-defined agent cards are available for:

- `DATA_LOADER_AGENT_CARD`: Data loading agent
- `DATA_ANALYST_AGENT_CARD`: Data analysis agent  
- `ORCHESTRATOR_AGENT_CARD`: Orchestration agent 