"""
Data Enrichment Agent Executor
This module contains the implementation of the Data Enrichment Agent's skills.
"""

import logging
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

logger = logging.getLogger(__name__)

class DataEnrichmentAgentExecutor:
    """
    Implements the logic for the data enrichment agent's skills.
    """
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        logger.info("DataEnrichmentAgentExecutor initialized.")

    async def enrich_dataset_skill(self, data_handle_id: str, operations: list = None) -> Dict[str, Any]:
        """
        A2A skill to enrich a dataset.
        """
        logger.info(f"Executing enrich_dataset_skill for data handle: {data_handle_id}")
        operations = operations or ["add_temporal_features"]

        try:
            handle = self.data_manager.get_handle(data_handle_id)
            if not handle:
                raise ValueError(f"Data handle not found: {data_handle_id}")

            # Get the actual data using the data manager
            df = self.data_manager.get_data(data_handle_id)
            operations = []
            enrichment_summary = {}

            if "add_temporal_features" in operations:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df['month'] = df['date'].dt.month
                    df['day_of_week'] = df['date'].dt.dayofweek
                    df['year'] = df['date'].dt.year
                    enrichment_summary['temporal_features_added'] = ['month', 'day_of_week', 'year']
            
            enriched_handle = self.data_manager.create_handle(
                data=df,
                data_type="dataframe",
                metadata={
                    "original_handle": data_handle_id,
                    "operations": operations,
                    "summary": enrichment_summary,
                }
            )

            return {
                "status": "completed",
                "enriched_data_handle_id": enriched_handle.handle_id,
                "summary": enrichment_summary,
            }
        except Exception as e:
            logger.exception(f"Error enriching dataset for handle {data_handle_id}: {e}")
            raise 