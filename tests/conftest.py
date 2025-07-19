import pytest
import asyncio

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.WindowsProactorEventLoopPolicy() if hasattr(asyncio, 'WindowsProactorEventLoopPolicy') else asyncio.DefaultEventLoopPolicy()

# Markers for different test types
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running") 