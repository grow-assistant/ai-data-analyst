# data_analyzer/analysis/trends.py

import pandas as pd
# Remove problematic imports and add local imports
# from data_analyzer.analysis.impact_analysis import analyze_periods
# from data_analyzer.analysis.timeframe import get_time_periods
import logging
from typing import List, Optional, Dict, Any
import io

# Import custom exceptions and types
try:
    from common_utils.types import AnalysisResult
    from common_utils.exceptions import (
        ColumnNotFoundError,
        InvalidDataError,
        InsufficientDataError,
        CalculationError,
        format_error_response
    )
except ImportError:
    # Fallback for when common_utils is not available
    class ColumnNotFoundError(Exception):
        pass
    class InvalidDataError(Exception):
        pass
    class InsufficientDataError(Exception):
        pass
    class CalculationError(Exception):
        pass
    def format_error_response(e):
        return {"error": str(e)}

logger = logging.getLogger(__name__)

def run_trend_analysis(df: pd.DataFrame, date_col: str, numerator_col: str, denominator_col: Optional[str], dimension_cols: List[str], timeframe: str, label: str, reference_date: Optional[pd.Timestamp] = None) -> Optional[Dict[str, Any]]:
    # Simplified version without complex dependencies
    if reference_date is None:
        reference_date = df[date_col].max()
    
    logger.debug(f"Trend Analysis for {label}: Timeframe {timeframe}")
    
    # Basic trend analysis
    result = {
        'label': label,
        'timeframe': timeframe,
        'numerator_col': numerator_col,
        'denominator_col': denominator_col,
        'dimension_cols': dimension_cols,
        'reference_date': reference_date
    }
    
    return result

def detect_trends(data: str, date_col: str, numerator_col: str, denominator_col: str = "", dimension_cols: str = "", timeframe: str = "wow") -> AnalysisResult:
    """
    ADK tool wrapper for trend detection analysis.
    
    Args:
        data: CSV string containing the dataset
        date_col: Name of the date column
        numerator_col: Name of the numerator column
        denominator_col: Name of the denominator column (optional)
        dimension_cols: Comma-separated list of dimension columns
        timeframe: Timeframe for analysis (wow, yoy, mom, etc.)
    
    Returns:
        AnalysisResult object with structured trend analysis
    """
    try:
        # Validate input data
        if not data or not data.strip():
            raise InvalidDataError("Data cannot be empty")
        
        # Parse CSV data
        try:
            df = pd.read_csv(io.StringIO(data))
        except Exception as e:
            raise InvalidDataError(f"Failed to parse CSV data: {str(e)}")
        
        # Check if dataframe has sufficient data
        if df.empty:
            raise InsufficientDataError("Dataset is empty", required_rows=1, actual_rows=0)
        
        if len(df) < 2:
            raise InsufficientDataError(
                "Trend analysis requires at least 2 data points", 
                required_rows=2, 
                actual_rows=len(df)
            )
        
        # Parse dimension columns
        dims = [col.strip() for col in dimension_cols.split(",") if col.strip()] if dimension_cols else []
        
        # Validate required columns exist
        available_columns = df.columns.tolist()
        
        if date_col not in df.columns:
            raise ColumnNotFoundError(date_col, available_columns)
        
        if numerator_col not in df.columns:
            raise ColumnNotFoundError(numerator_col, available_columns)
        
        # Validate optional denominator column
        if denominator_col and denominator_col not in df.columns:
            raise ColumnNotFoundError(denominator_col, available_columns)
        
        # Validate dimension columns
        for dim in dims:
            if dim not in df.columns:
                raise ColumnNotFoundError(dim, available_columns)
        
        # Convert date column
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise InvalidDataError(f"Failed to parse date column '{date_col}': {str(e)}")
        
        # Perform basic trend analysis
        results = {}
        summary_lines = []
        
        try:
            # Basic statistics
            results['dataset_shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
            results['parameters'] = {
                'date_col': date_col,
                'numerator_col': numerator_col,
                'denominator_col': denominator_col,
                'timeframe': timeframe,
                'dimensions': dims
            }
            summary_lines.append(f"Trend Analysis Results for '{numerator_col}':")
            
            # Calculate statistics for numerator column
            if numerator_col in df.columns:
                numerator_mean = df[numerator_col].mean()
                numerator_diff_mean = df[numerator_col].diff().mean()
                numerator_trend = 'Increasing' if numerator_diff_mean > 0 else 'Decreasing' if numerator_diff_mean < 0 else 'Stable'

                results['numerator_stats'] = {
                    'mean': numerator_mean,
                    'trend': numerator_trend,
                    'diff_mean': numerator_diff_mean
                }
                summary_lines.append(f"  - Overall trend is {numerator_trend.lower()} (avg change: {numerator_diff_mean:.2f}).")
                
                # Additional trend metrics
                if len(df) >= 3:
                    recent_trend_mean = df[numerator_col].iloc[-3:].diff().mean()
                    recent_trend_direction = 'Accelerating' if recent_trend_mean > 0 else 'Decelerating'
                    results['recent_trend'] = {
                        'mean_diff': recent_trend_mean,
                        'direction': recent_trend_direction
                    }
                    summary_lines.append(f"  - Recent trend (last 3 points) is {recent_trend_direction.lower()}.")

        except Exception as e:
            raise CalculationError("trend analysis", str(e))
        
        # Construct the final AnalysisResult
        return AnalysisResult(
            analysis_type="trend_detection",
            results=results,
            visualization_data=None, # Can be added later
            summary="\n".join(summary_lines)
        )
        
    except (ColumnNotFoundError, InvalidDataError, InsufficientDataError, CalculationError) as e:
        # Return a structured error response
        error_info = format_error_response(e)
        return AnalysisResult(
            analysis_type="trend_detection_error",
            results=error_info,
            visualization_data=None,
            summary=f"Error in trend analysis: {error_info.get('message', str(e))}"
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in trend analysis: {e}")
        error_info = format_error_response(e)
        return AnalysisResult(
            analysis_type="trend_detection_error",
            results=error_info,
            visualization_data=None,
            summary=f"An unexpected error occurred: {str(e)}"
        )
