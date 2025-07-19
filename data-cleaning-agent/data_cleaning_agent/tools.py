from common_utils.mcp_server.tool_server import BaseTool
from common_utils.data_handle_manager import get_data_handle_manager
import pandas as pd
from typing import Dict, Any, List

class CleaningTool(BaseTool):
    name = "clean_dataset"
    description = "Cleans a dataset by removing duplicates and handling missing values."

    def __init__(self):
        super().__init__()
        self.data_manager = get_data_handle_manager()

    async def execute(self, data_handle_id: str, operations: List[str] = None) -> Dict[str, Any]:
        """
        Cleans the dataset identified by the data_handle_id.

        :param data_handle_id: The ID of the data handle for the dataset to clean.
        :param operations: A list of cleaning operations to perform.
                           Defaults to ["remove_duplicates", "handle_missing_values"].
        :return: A dictionary containing the results of the cleaning operation.
        """
        operations = operations or ["remove_duplicates", "handle_missing_values"]

        handle = self.data_manager.get_handle(data_handle_id)
        if not handle:
            raise ValueError(f"Data handle not found: {data_handle_id}")

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
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Impute with mode, handle potential for multiple modes
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col].fillna(mode_val[0], inplace=True)
            cleaning_summary['missing_values_filled'] = int(initial_nulls - df.isnull().sum().sum())

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

        # Helper to convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (pd.Timestamp,)):
                 return obj.isoformat()
            if hasattr(obj, 'dtype'):
                return obj.item() if hasattr(obj, 'item') else str(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_numpy(i) for i in obj]
            return obj

        return {
            "status": "completed",
            "cleaned_data_handle_id": cleaned_handle.handle_id,
            "original_shape": convert_numpy(original_shape),
            "cleaned_shape": convert_numpy(df.shape),
            "operations": operations,
            "summary": convert_numpy(cleaning_summary),
            "message": f"Successfully cleaned dataset with {len(operations)} operations."
        } 