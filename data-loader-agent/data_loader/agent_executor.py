"""
Enhanced Data Loader Agent Executor with Tableau Hyper API Support
This module provides fast loading for TDSX, Hyper, and other data formats.
"""

import logging
import os
import pandas as pd
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import numpy as np

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common_utils.data_handle_manager import get_data_handle_manager

# Tableau Hyper API imports
try:
    from tableauhyperapi import HyperProcess, Connection, Telemetry, CreateMode
    from tableauhyperapi import TableDefinition, SqlType, escape_name
    from tableauhyperapi import inserter
    HYPER_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Tableau Hyper API available - enhanced TDSX/Hyper support enabled")
except ImportError:
    HYPER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Tableau Hyper API not available - falling back to basic TDSX extraction")

logger = logging.getLogger(__name__)

def read_hyper_file(file_path: Path) -> pd.DataFrame:
    """
    Read a Tableau Hyper file using the Hyper API for optimal performance.
    Reference: https://tableau.github.io/hyper-db/docs/
    """
    if not HYPER_AVAILABLE:
        raise ImportError("Tableau Hyper API not available. Install with: pip install tableauhyperapi")
    
    logger.info(f"Reading Hyper file with Tableau Hyper API: {file_path}")
    
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=str(file_path)) as connection:
            # Get all table names in the database
            table_names = connection.execute_list_query(
                "SELECT table_name FROM information_schema.tables WHERE table_type='TABLE'"
            )
            
            if not table_names:
                raise ValueError(f"No tables found in Hyper file: {file_path}")
            
            # Use the first table (or you could iterate through all)
            table_name = table_names[0][0]
            logger.info(f"Reading table: {table_name}")
            
            # Query the table and convert to pandas DataFrame
            query = f"SELECT * FROM {escape_name(table_name)}"
            result = connection.execute_query(query)
            
            # Get column names
            column_names = [column.name for column in result.schema]
            
            # Fetch all rows
            rows = []
            for row in result:
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=column_names)
            
            logger.info(f"Successfully loaded {len(df)} rows from Hyper file")
            return df

def read_tdsx_with_hyper(file_path: Path) -> pd.DataFrame:
    """
    Extract and read TDSX files using Hyper API for data extraction.
    TDSX files may contain .hyper files internally.
    """
    import zipfile
    import tempfile
    
    logger.info(f"Processing TDSX file with Hyper API support: {file_path}")
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        # Look for Hyper files first (fastest)
        hyper_files = [f for f in file_list if f.endswith('.hyper')]
        
        if hyper_files and HYPER_AVAILABLE:
            # Extract and read the first Hyper file
            hyper_file = hyper_files[0]
            logger.info(f"Found Hyper file in TDSX: {hyper_file}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_path = Path(temp_dir) / hyper_file
                zip_ref.extract(hyper_file, temp_dir)
                return read_hyper_file(extracted_path)
        
        else:
            # Fall back to CSV/TSV extraction
            logger.info("No Hyper files found or Hyper API unavailable, using CSV extraction")
            return read_tdsx_dataframe_fallback(file_path, zip_ref, file_list)

def read_tdsx_dataframe_fallback(file_path: Path, zip_ref=None, file_list=None) -> pd.DataFrame:
    """
    Fallback method for reading TDSX files without Hyper API.
    """
    import zipfile
    
    if zip_ref is None:
        zip_ref = zipfile.ZipFile(file_path, 'r')
        file_list = zip_ref.namelist()
    
    # Look for CSV or TSV data files
    data_files = [f for f in file_list if f.endswith(('.csv', '.txt', '.tsv'))]
    
    if data_files:
        # Extract and read the first data file found
        data_file = data_files[0]
        logger.info(f"Reading data file from TDSX: {data_file}")
        
        with zip_ref.open(data_file) as data_content:
            # Enhanced encoding detection
            import chardet
            
            # Read a sample to detect encoding
            sample = data_content.read(10000)
            data_content.seek(0)
            
            detected = chardet.detect(sample)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0)
            
            logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Try reading with detected encoding
            encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for enc in encodings_to_try:
                try:
                    data_content.seek(0)
                    
                    if data_file.endswith('.csv'):
                        df = pd.read_csv(data_content, encoding=enc)
                    elif data_file.endswith('.tsv'):
                        df = pd.read_csv(data_content, sep='\t', encoding=enc)
                    else:
                        # Try comma first, then tab
                        try:
                            data_content.seek(0)
                            df = pd.read_csv(data_content, encoding=enc)
                        except:
                            data_content.seek(0)
                            df = pd.read_csv(data_content, sep='\t', encoding=enc)
                    
                    logger.info(f"Successfully read TDSX data with encoding: {enc}")
                    return df
                    
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    logger.warning(f"Failed to read with encoding {enc}: {e}")
                    continue
            
            raise ValueError(f"Unable to read data file {data_file} with any encoding")
    else:
        raise ValueError(f"No data files found in TDSX: {file_path}")

# Legacy function for backward compatibility
def read_tdsx_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Read a TDSX file - enhanced with Hyper API support.
    """
    try:
        return read_tdsx_with_hyper(file_path)
    except Exception as e:
        logger.warning(f"Hyper API read failed, falling back to basic extraction: {e}")
        return read_tdsx_dataframe_fallback(file_path)

class EnhancedDataLoaderExecutor:
    """
    Enhanced Data Loader Agent with Tableau Hyper API support.
    Supports fast loading of TDSX, Hyper, CSV, JSON, Excel and other formats.
    """
    
    def __init__(self):
        self.data_manager = get_data_handle_manager()
        logger.info("Enhanced Data Loader Agent initialized")
        if HYPER_AVAILABLE:
            logger.info("✅ Tableau Hyper API support enabled for fast TDSX/Hyper file reading")
        else:
            logger.info("⚠️ Tableau Hyper API not available - using fallback methods")

    async def load_data_skill(self, file_path: str, file_type: str = "auto") -> Dict[str, Any]:
        """
        Enhanced A2A skill to load data from various file formats including Hyper.
        """
        logger.info(f"Loading data from: {file_path} (type: {file_type})")

        try:
            # Convert to Path object
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Auto-detect file type if not specified
            if file_type == "auto":
                file_type = self._detect_file_type(path)
            
            logger.info(f"Detected/specified file type: {file_type}")

            # Load data based on file type
            if file_type == "hyper":
                df = read_hyper_file(path)
                metadata = {
                    "source_file": str(path),
                    "file_type": "hyper",
                    "loader_method": "tableau_hyper_api",
                    "fast_loading": True
                }
            elif file_type == "tdsx":
                df = read_tdsx_with_hyper(path)
                metadata = {
                    "source_file": str(path),
                    "file_type": "tdsx",
                    "loader_method": "hyper_api_enhanced" if HYPER_AVAILABLE else "fallback_extraction",
                    "fast_loading": HYPER_AVAILABLE
                }
            elif file_type == "csv":
                df = self._read_csv_enhanced(path)
                metadata = {"source_file": str(path), "file_type": "csv"}
            elif file_type == "json":
                df = pd.read_json(path)
                metadata = {"source_file": str(path), "file_type": "json"}
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(path)
                metadata = {"source_file": str(path), "file_type": file_type}
            elif file_type == "tsv":
                df = self._read_tsv_enhanced(path)
                metadata = {"source_file": str(path), "file_type": "tsv"}
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Validate loaded data
            if df.empty:
                raise ValueError("Loaded dataset is empty")

            # Convert numpy types to native Python types for JSON serialization
            df = self._convert_numpy_types(df)
            
            # Add loading statistics to metadata
            metadata.update({
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "hyper_api_available": HYPER_AVAILABLE
            })

            # Create data handle
            data_handle = self.data_manager.create_handle(
                data=df,
                data_type="dataframe",
                metadata=metadata
            )

            logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Data handle created: {data_handle.handle_id}")

            return {
                "status": "completed",
                "data_handle_id": data_handle.handle_id,
                "metadata": metadata,
                "summary": {
                    "rows_loaded": len(df),
                    "columns_loaded": len(df.columns),
                    "file_size_mb": path.stat().st_size / (1024 * 1024),
                    "fast_loading_used": metadata.get("fast_loading", False)
                }
            }

        except Exception as e:
            logger.exception(f"Error loading data from {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "file_type": file_type
            }

    def _detect_file_type(self, path: Path) -> str:
        """Enhanced file type detection."""
        suffix = path.suffix.lower()
        
        # Direct mappings
        type_map = {
            '.hyper': 'hyper',
            '.tdsx': 'tdsx',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.json': 'json',
            '.xlsx': 'xlsx',
            '.xls': 'xls',
            '.txt': 'csv'  # Assume CSV for .txt files
        }
        
        detected_type = type_map.get(suffix, 'csv')
        logger.info(f"Auto-detected file type: {detected_type} for {path}")
        return detected_type

    def _read_csv_enhanced(self, path: Path) -> pd.DataFrame:
        """Enhanced CSV reading with encoding detection."""
        import chardet
        
        # Detect encoding
        with open(path, 'rb') as f:
            sample = f.read(10000)
            detected = chardet.detect(sample)
            encoding = detected.get('encoding', 'utf-8')
            confidence = detected.get('confidence', 0)
        
        logger.info(f"CSV encoding detection: {encoding} (confidence: {confidence:.2f})")
        
        # Try multiple encodings
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                logger.info(f"Successfully read CSV with encoding: {enc}")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        raise ValueError(f"Unable to read CSV file with any encoding: {path}")

    def _read_tsv_enhanced(self, path: Path) -> pd.DataFrame:
        """Enhanced TSV reading with encoding detection."""
        import chardet
        
        # Detect encoding
        with open(path, 'rb') as f:
            sample = f.read(10000)
            detected = chardet.detect(sample)
            encoding = detected.get('encoding', 'utf-8')
        
        # Try multiple encodings
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings:
            try:
                df = pd.read_csv(path, sep='\t', encoding=enc)
                logger.info(f"Successfully read TSV with encoding: {enc}")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        raise ValueError(f"Unable to read TSV file with any encoding: {path}")

    def _convert_numpy_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numpy types to native Python types for JSON serialization."""
        for col in df.columns:
            if df[col].dtype == 'object':
                continue
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype('Int64')  # Nullable integer
            elif pd.api.types.is_float_dtype(df[col]):
                # Keep as float64, pandas handles this well
                pass
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Keep as datetime, pandas handles this
                pass
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype('boolean')  # Nullable boolean
        
        return df

# Maintain backward compatibility
DataLoaderAgentExecutor = EnhancedDataLoaderExecutor 