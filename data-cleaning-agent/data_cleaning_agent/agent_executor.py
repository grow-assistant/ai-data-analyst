"""
Data Cleaning Agent Executor
This module contains the implementation of the Data Cleaning Agent's skills.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

logger = logging.getLogger(__name__)

class DataCleaningAgentExecutor:
    """
    Implements the logic for the data cleaning agent's skills.
    """
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        logger.info("DataCleaningAgentExecutor initialized.")

    async def clean_dataset_skill(self, data_handle_id: str, operations: list = None) -> Dict[str, Any]:
        """
        A2A skill to clean a dataset.
        """
        logger.info(f"Executing clean_dataset_skill for data handle: {data_handle_id}")
        operations = operations or ["remove_duplicates", "handle_missing_values"]

        try:
            handle = self.data_manager.get_handle(data_handle_id)
            if not handle:
                raise ValueError(f"Data handle not found: {data_handle_id}")

            # Get the actual data using the data manager
            df = self.data_manager.get_data(data_handle_id)
            original_shape = df.shape
            cleaning_summary = {}

            if "remove_duplicates" in operations:
                initial_rows = len(df)
                df.drop_duplicates(inplace=True)
                cleaning_summary['duplicates_removed'] = initial_rows - len(df)

            if "handle_missing_values" in operations:
                initial_nulls = df.isnull().sum().sum()
                for col in df.columns:
                    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                cleaning_summary['missing_values_filled'] = initial_nulls - df.isnull().sum().sum()

            cleaned_handle = self.data_manager.create_handle(
                data=df,
                data_type="dataframe",
                metadata={
                    "original_handle": data_handle_id,
                    "original_shape": original_shape,
                    "cleaned_shape": df.shape,
                    "operations": operations,
                    "summary": cleaning_summary,
                }
            )

            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'dtype'):  # numpy types
                    return obj.item() if hasattr(obj, 'item') else str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            return {
                "status": "completed",
                "cleaned_data_handle_id": cleaned_handle.handle_id,
                "original_shape": convert_numpy_types(original_shape),
                "cleaned_shape": convert_numpy_types(df.shape),
                "operations": operations,
                "summary": convert_numpy_types(cleaning_summary),
                "message": f"Successfully cleaned dataset with {len(operations)} operations."
            }
        except Exception as e:
            logger.exception(f"Error cleaning dataset for handle {data_handle_id}: {e}")
            raise 