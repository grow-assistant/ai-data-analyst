import pandas as pd
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

def get_reference_date_from_config(config: Dict[str, Any], df: pd.DataFrame, date_col: str) -> pd.Timestamp:
    """
    Get the reference date from config, with fallback to max date in data.
    Also filters the dataframe if a reference date is configured.
    
    Returns:
        Tuple of (reference_date, filtered_dataframe)
    """
    analysis_config = config.get('analysis', {})
    reference_date_str = analysis_config.get('reference_date')
    
    if reference_date_str:
        try:
            reference_date = pd.to_datetime(reference_date_str)
            logger.info(f"Using configured reference date: {reference_date.strftime('%Y-%m-%d')}")
            return reference_date
        except Exception as e:
            logger.warning(f"Invalid reference_date format '{reference_date_str}': {e}. Using max date from data.")
    
    return df[date_col].max()

def get_time_periods(timeframe: str, reference_date: Optional[pd.Timestamp] = None) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Given a timeframe and a reference date, return a dictionary with:
    - current: (start_date, end_date)
    - prior: (start_date, end_date)
    - yoy: (start_date, end_date)

    Timeframes:
    - "last_week"
    - "month_to_date"
    - "last_month"
    - "quarter_to_date"
    - "last_quarter"
    - "year_to_date"
    - "last_year"
    - "rolling_4_weeks"
    - "rolling_2_weeks"

    The reference_date should be a Timestamp (usually max date in data).
    If not provided, defaults to today.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    timeframe = timeframe.lower().replace(" ", "_")

    if timeframe == "last_week":
        # Last completed week (Mon-Sun or Sun-Sat; here we assume Sunday-end)
        # We'll take the most recently completed Sunday as end_date.
        end_of_last_week = reference_date - pd.to_timedelta(reference_date.weekday() + 1, unit='D')
        start_of_last_week = end_of_last_week - pd.Timedelta(6, unit='D')

        current_start, current_end = start_of_last_week, end_of_last_week
        # Prior week
        prior_start = current_start - pd.Timedelta(7, unit='D')
        prior_end = current_start - pd.Timedelta(1, unit='D')
        # YOY same week last year
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = current_end - pd.DateOffset(years=1)

    elif timeframe == "month_to_date":
        # Current MTD
        current_start = reference_date.replace(day=1)
        current_end = reference_date
        # Prior period: same number of days from the previous month
        prior_month_end = current_start - pd.Timedelta(1, unit='D')
        prior_month_start = prior_month_end.replace(day=1)
        prior_start, prior_end = prior_month_start, prior_month_start + (current_end - current_start)
        if prior_end > prior_month_end:
            prior_end = prior_month_end
        # YOY: same period last year
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = yoy_start + (current_end - current_start)

    elif timeframe == "last_month":
        # Entire last month
        first_of_current_month = reference_date.replace(day=1)
        last_month_end = first_of_current_month - pd.Timedelta(1, unit='D')
        last_month_start = last_month_end.replace(day=1)

        current_start, current_end = last_month_start, last_month_end
        # Prior month
        prior_month_end = current_start - pd.Timedelta(1, unit='D')
        prior_month_start = prior_month_end.replace(day=1)
        prior_start, prior_end = prior_month_start, prior_month_end
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = current_end - pd.DateOffset(years=1)

    elif timeframe == "quarter_to_date":
        # Determine current quarter start
        current_quarter = (reference_date.month - 1) // 3 + 1
        quarter_start_month = 3 * (current_quarter - 1) + 1
        current_start = reference_date.replace(month=quarter_start_month, day=1)
        current_end = reference_date
        # Prior quarter period: same length in previous quarter
        prev_quarter_end = current_start - pd.Timedelta(1, unit='D')
        prev_quarter_start = (prev_quarter_end - QuarterEnd(startingMonth=1)) + pd.Timedelta(1, unit='D')
        # Match the length of the current QTD range
        period_length = current_end - current_start
        prior_start = prev_quarter_start
        prior_end = prior_start + period_length
        if prior_end > prev_quarter_end:
            prior_end = prev_quarter_end
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = yoy_start + (current_end - current_start)

    elif timeframe == "last_quarter":
        # Identify last quarter
        current_quarter = (reference_date.month - 1) // 3 + 1
        last_quarter_end = (reference_date.replace(month=3 * (current_quarter - 1) + 1, day=1) - pd.Timedelta(1, unit='D'))
        last_quarter_start = (last_quarter_end - QuarterEnd(startingMonth=1)) + pd.Timedelta(1, unit='D')

        current_start, current_end = last_quarter_start, last_quarter_end
        # Prior quarter
        prior_quarter_end = current_start - pd.Timedelta(1, unit='D')
        prior_quarter_start = (prior_quarter_end - QuarterEnd(startingMonth=1)) + pd.Timedelta(1, unit='D')
        prior_start, prior_end = prior_quarter_start, prior_quarter_end
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = current_end - pd.DateOffset(years=1)

    elif timeframe == "year_to_date":
        current_start = reference_date.replace(month=1, day=1)
        current_end = reference_date
        period_length = current_end - current_start
        # Prior period (same length ending just before current_start)
        prior_end = current_start - pd.Timedelta(1, unit='D')
        prior_start = prior_end - period_length
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = yoy_start + period_length

    elif timeframe == "last_year":
        # Entire last year
        current_year_start = reference_date.replace(month=1, day=1)
        last_year_end = current_year_start - pd.Timedelta(1, unit='D')
        last_year_start = last_year_end.replace(month=1, day=1)

        current_start, current_end = last_year_start, last_year_end
        # Prior year
        prior_year_end = current_start - pd.Timedelta(1, unit='D')
        prior_year_start = prior_year_end.replace(month=1, day=1)
        prior_start, prior_end = prior_year_start, prior_year_end
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = current_end - pd.DateOffset(years=1)

    elif timeframe == "rolling_4_weeks":
        # Rolling 4 full weeks from the reference_date (28 days)
        # Assume reference_date is the end of a reporting period, otherwise adjust logic
        current_end = reference_date
        current_start = current_end - pd.Timedelta(27, unit='D')  # 4 weeks = 28 days, but end-inclusive
        # Prior 4 weeks
        prior_end = current_start - pd.Timedelta(1, unit='D')
        prior_start = prior_end - pd.Timedelta(27, unit='D')
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = yoy_start + (current_end - current_start)

    elif timeframe == "rolling_2_weeks":
        # Rolling 2 weeks comparison: current 2 weeks vs prior 2 weeks
        current_end = reference_date
        current_start = current_end - pd.Timedelta(13, unit='D')  # 2 weeks = 14 days, but end-inclusive
        # Prior 2 weeks (the 2 weeks before current period)
        prior_end = current_start - pd.Timedelta(1, unit='D')
        prior_start = prior_end - pd.Timedelta(13, unit='D')  # 2 weeks = 14 days, but end-inclusive
        # YOY
        yoy_start = current_start - pd.DateOffset(years=1)
        yoy_end = yoy_start + (current_end - current_start)

    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    return {
        'current': (current_start.normalize(), current_end.normalize()),
        'prior': (prior_start.normalize(), prior_end.normalize()),
        'yoy': (yoy_start.normalize(), yoy_end.normalize())
    }

def analyze_timeframe(data: str, date_col: str, value_col: str, timeframe: str = "weekly") -> str:
    """
    ADK tool wrapper for timeframe analysis.
    
    Args:
        data: CSV string containing the dataset
        date_col: Name of the date column
        value_col: Name of the value column
        timeframe: Timeframe for analysis (daily, weekly, monthly, quarterly, yearly)
    
    Returns:
        String summary of the timeframe analysis
    """
    try:
        import io
        
        df = pd.read_csv(io.StringIO(data))
        
        if date_col not in df.columns:
            return f"Error: Date column '{date_col}' not found in data"
        if value_col not in df.columns:
            return f"Error: Value column '{value_col}' not found in data"
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set up timeframe grouping
        if timeframe == "daily":
            df['period'] = df[date_col].dt.date
        elif timeframe == "weekly":
            df['period'] = df[date_col].dt.to_period('W')
        elif timeframe == "monthly":
            df['period'] = df[date_col].dt.to_period('M')
        elif timeframe == "quarterly":
            df['period'] = df[date_col].dt.to_period('Q')
        elif timeframe == "yearly":
            df['period'] = df[date_col].dt.to_period('Y')
        else:
            return f"Error: Unsupported timeframe '{timeframe}'"
        
        # Group by period
        period_summary = df.groupby('period')[value_col].agg(['mean', 'sum', 'count', 'std'])
        
        result_summary = f"Timeframe Analysis Results\n"
        result_summary += f"Date Column: {date_col}\n"
        result_summary += f"Value Column: {value_col}\n"
        result_summary += f"Timeframe: {timeframe}\n\n"
        
        result_summary += f"Period Summary (showing first 10):\n"
        for period, row in period_summary.head(10).iterrows():
            result_summary += f"  {period}: Mean={row['mean']:.4f}, Sum={row['sum']:.4f}, Count={row['count']}, Std={row['std']:.4f}\n"
        
        # Overall statistics
        result_summary += f"\nOverall Statistics:\n"
        result_summary += f"  Total periods: {len(period_summary)}\n"
        result_summary += f"  Average per period: {period_summary['mean'].mean():.4f}\n"
        result_summary += f"  Period-to-period volatility: {period_summary['mean'].std():.4f}\n"
        
        return result_summary
        
    except Exception as e:
        return f"Error in timeframe analysis: {str(e)}"
