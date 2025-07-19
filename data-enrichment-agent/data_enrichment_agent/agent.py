"""
Data Enrichment Agent
This module defines the top-level agent components.
The agent's business logic is implemented in the `DataEnrichmentAgentExecutor`.
"""

import logging

logger = logging.getLogger(__name__)

# The logic that was previously in this file has been moved to
# `agent_executor.py` to better align with the a2a-sdk structure.

# The `__main__.py` script now directly uses the `DataEnrichmentAgentExecutor`
# to create the A2A server and its skills. This file is kept for
# package structure but is no longer the primary implementation file.

logger.info("Data Enrichment Agent package loaded. See agent_executor.py for implementation.")
