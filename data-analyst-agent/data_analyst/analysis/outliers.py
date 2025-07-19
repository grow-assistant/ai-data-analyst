# data_analyzer/analysis/outliers.py
import pandas as pd

# Import AnalysisResult and exceptions
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
    class AnalysisResult:
        def __init__(self, analysis_type, results, visualization_data, summary):
            self.analysis_type = analysis_type
            self.results = results
            self.visualization_data = visualization_data
            self.summary = summary
    
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

def run_outlier_analysis(df: pd.DataFrame, metric_col: str, record_id_col: str, threshold_factor: float = 3.0) -> pd.DataFrame:
    """
    Detects record-level outliers in a given metric column of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the metric data.
        metric_col (str): The name of the metric column in df.
        record_id_col (str): The unique record identifier column in df.
        threshold_factor (float): The factor used to determine outliers. 
                                  Outliers are defined as values exceeding mean +/- threshold_factor * std.

    Returns:
        pd.DataFrame: A DataFrame containing outlier records with their metric values and deviation info.
    """
    if metric_col not in df.columns or record_id_col not in df.columns:
        raise ValueError("Specified columns do not exist in the DataFrame.")

    metric_values = df[metric_col]
    mean_val = metric_values.mean()
    std_val = metric_values.std()

    # Define outlier thresholds
    upper_threshold = mean_val + threshold_factor * std_val
    lower_threshold = mean_val - threshold_factor * std_val

    # Filter outliers
    outliers = df[(df[metric_col] > upper_threshold) | (df[metric_col] < lower_threshold)].copy()
    if not outliers.empty:
        outliers['mean'] = mean_val
        outliers['std'] = std_val
        outliers['z_score'] = (outliers[metric_col] - mean_val) / std_val

    return outliers

def is_member_an_outlier(df: pd.DataFrame, dimension_col: str, member_name: str, metric_col: str, threshold_factor: float = 2.0) -> bool:
    """
    Determines if a specific member of a dimension is an outlier compared to its peers.
    """
    if dimension_col not in df.columns or metric_col not in df.columns:
        return False
        
    # Aggregate metric by dimension to get peer performance
    peer_group = df.groupby(dimension_col)[metric_col].mean()

    if member_name not in peer_group.index:
        return False

    member_value = peer_group.loc[member_name]
    
    # Exclude the member itself for a fair comparison
    other_peers = peer_group.drop(member_name)
    
    if other_peers.empty:
        return False # Cannot determine outlier status without peers

    mean_val = other_peers.mean()
    std_val = other_peers.std()

    # If standard deviation is zero, no outliers can be detected
    if std_val == 0:
        return False

    # Define outlier thresholds
    upper_threshold = mean_val + threshold_factor * std_val
    lower_threshold = mean_val - threshold_factor * std_val

    return member_value > upper_threshold or member_value < lower_threshold

def detect_outliers(data: str, column: str, method: str = "iqr", threshold: float = 1.5) -> AnalysisResult:
    """
    ADK tool wrapper for outlier detection analysis.
    
    Args:
        data: CSV string containing the dataset
        column: Name of the column to analyze for outliers
        method: Method for outlier detection ("iqr", "zscore", "isolation")
        threshold: Threshold for outlier detection
    
    Returns:
        AnalysisResult object with structured outlier detection results
    """
    try:
        import io
        import numpy as np
        from scipy import stats
        
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
        
        # Validate column exists
        available_columns = df.columns.tolist()
        if column not in df.columns:
            raise ColumnNotFoundError(column, available_columns)
        
        # Extract and clean column data
        col_data = df[column].dropna()
        
        if len(col_data) < 3:
            raise InsufficientDataError(
                "Outlier detection requires at least 3 data points", 
                required_rows=3, 
                actual_rows=len(col_data)
            )
        
        # Initialize results structure
        results = {
            'parameters': {
                'column': column,
                'method': method,
                'threshold': threshold,
                'total_values': len(df[column]),
                'valid_values': len(col_data)
            },
            'statistics': {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            }
        }
        
        summary_lines = [f"Outlier Detection Analysis for '{column}':"]
        
        # Perform outlier detection based on method
        try:
            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                results['method_details'] = {
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
                summary_lines.append(f"  - Method: IQR with threshold {threshold}")
                summary_lines.append(f"  - Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > threshold
                outliers = col_data[outlier_mask]
                outlier_z_scores = z_scores[outlier_mask]
                
                results['method_details'] = {
                    'threshold_z_score': threshold,
                    'max_z_score': float(z_scores.max()),
                    'outlier_z_scores': outlier_z_scores.tolist()
                }
                
                summary_lines.append(f"  - Method: Z-Score with threshold {threshold}")
                summary_lines.append(f"  - Max Z-score in data: {z_scores.max():.2f}")
                
            else:
                raise InvalidDataError(f"Unsupported method '{method}'. Use 'iqr' or 'zscore'.")
            
            # Store outlier results
            results['outliers'] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(col_data)) * 100,
                'values': outliers.tolist()[:20],  # Limit to first 20 outliers
                'severity': 'high' if len(outliers) > len(col_data) * 0.1 else 'moderate' if len(outliers) > 0 else 'none'
            }
            
            # Generate summary
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(col_data)) * 100
            
            summary_lines.append(f"  - Found {outlier_count} outliers ({outlier_percentage:.1f}% of data)")
            
            if outlier_count > 0:
                summary_lines.append(f"  - Outlier severity: {results['outliers']['severity']}")
                if outlier_count <= 5:
                    summary_lines.append(f"  - Outlier values: {outliers.tolist()}")
                else:
                    summary_lines.append(f"  - Sample outlier values: {outliers.tolist()[:5]}...")
            else:
                summary_lines.append("  - No outliers detected in the data")
                
        except Exception as e:
            raise CalculationError("outlier detection", str(e))
        
        # Construct the final AnalysisResult
        return AnalysisResult(
            analysis_type="outlier_detection",
            results=results,
            visualization_data=None,  # Can be added later for plotting
            summary="\n".join(summary_lines)
        )
        
    except (ColumnNotFoundError, InvalidDataError, InsufficientDataError, CalculationError) as e:
        # Return a structured error response
        error_info = format_error_response(e)
        return AnalysisResult(
            analysis_type="outlier_detection_error",
            results=error_info,
            visualization_data=None,
            summary=f"Error in outlier detection: {error_info.get('message', str(e))}"
        )
    except Exception as e:
        # Handle unexpected errors
        error_info = format_error_response(e)
        return AnalysisResult(
            analysis_type="outlier_detection_error",
            results=error_info,
            visualization_data=None,
            summary=f"An unexpected error occurred: {str(e)}"
        )
