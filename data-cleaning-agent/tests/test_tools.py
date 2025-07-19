import pytest
import pandas as pd
import numpy as np
from data_cleaning_agent.tools import handle_missing_values, remove_outliers, standardize_date_column

@pytest.fixture
def sample_dataframe():
    data = {
        'Sales': [100, 200, np.nan, 150, 300],
        'Revenue': [1000, 2000, 1500, 1200, 10000],
        'OrderDate': ['2023-01-05', '2023/02/10', '2023-03-15', '2023-04-20', '2023-05-25']
    }
    return pd.DataFrame(data)

def test_handle_missing_values_mean(sample_dataframe):
    cleaned_df = handle_missing_values(sample_dataframe, strategy='mean', subset=['Sales'])
    assert not cleaned_df['Sales'].isnull().any()
    assert cleaned_df['Sales'].iloc[2] == pytest.approx(187.5, rel=1e-2)  # Mean of [100, 200, 150, 300]

def test_remove_outliers_iqr(sample_dataframe):
    cleaned_df = remove_outliers(sample_dataframe, method='iqr', subset=['Revenue'])
    assert len(cleaned_df) < len(sample_dataframe)  # Outlier should be removed
    assert 10000 not in cleaned_df['Revenue'].values

def test_standardize_date_column(sample_dataframe):
    cleaned_df = standardize_date_column(sample_dataframe, column_name='OrderDate', date_format='%Y-%m-%d')
    assert cleaned_df['OrderDate'].iloc[0] == '2023-01-05'
    assert cleaned_df['OrderDate'].iloc[1] == '2023-02-10'  # Assuming the function standardizes the format 