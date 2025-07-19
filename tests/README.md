# Integration Tests for ADK Multi-Agent System

This directory contains integration tests for the ADK-based multi-agent system that test actual running agents.

## Test Structure

- `integration/` - Integration tests that start and test real agent processes
- `setup_test_env.py` - Environment setup for tests
- `requirements.txt` - Test dependencies
- `conftest.py` - Pytest configuration
- `pytest.ini` - Pytest settings

## Test Files

### `test_simple_startup.py`
Basic test that verifies the MCP server can start and become available. Good for debugging agent startup issues.

### `test_agents.py`
Full integration test suite that:
1. Starts all agents (MCP server, Data Loader, Data Analyst, Orchestrator)
2. Tests health endpoints
3. Tests MCP routing functionality
4. Tests agent registration
5. Tests orchestrator workflows

## How to Run Tests

### Prerequisites
1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio httpx
   ```

2. Ensure you have a valid Google API key (the tests use a dummy key for basic functionality)

### Running Tests

1. **Simple startup test** (recommended first):
   ```bash
   python -m pytest tests/integration/test_simple_startup.py -v -s
   ```

2. **Full integration test suite**:
   ```bash
   python -m pytest tests/integration/test_agents.py -v -s
   ```

3. **All tests**:
   ```bash
   python -m pytest tests/ -v -s
   ```

4. **Using the test runner**:
   ```bash
   python tests/run_integration_tests.py
   ```

## Test Configuration

The tests automatically:
- Set up required environment variables
- Start agents in the correct order (MCP server first)
- Wait for each agent to become available
- Clean up processes after tests complete

## Ports Used

- MCP Server: 10001
- Orchestrator Agent: 10000
- Data Loader Agent: 10006
- Data Analyst Agent: 10007

## Troubleshooting

### Common Issues

1. **Port conflicts**: Make sure no other services are using the required ports
2. **Startup timeouts**: Agents may take time to start, especially on slower systems
3. **API key errors**: While tests use dummy keys, some functionality may require valid keys

### Debug Tips

1. Run tests with `-s` flag to see print output
2. Check agent logs in the test output
3. Use the simple startup test first to isolate issues
4. Ensure all dependencies are installed

## Test Coverage

The integration tests cover:
- ✅ Agent startup and availability
- ✅ Health endpoint functionality  
- ✅ MCP server routing
- ✅ Agent registration with MCP
- ✅ Basic orchestrator workflows
- ⚠️ Full end-to-end data workflows (may require valid API keys)

## Future Enhancements

- Add tests for error handling scenarios
- Test with real data files
- Add performance/load testing
- Test agent failure and recovery scenarios
- Add tests for security features 