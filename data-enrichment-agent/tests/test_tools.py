import pytest
import pandas as pd
import requests
from unittest.mock import patch
from data_enrichment_agent.tools import fetch_external_data, merge_datasets, calculate_moving_average

@pytest.fixture
def sample_dataframe():
    data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Sales': [100, 150, 120, 180]
    }
    return pd.DataFrame(data)

@pytest.fixture
def external_dataframe():
    data = {
        'Date': ['2023-01-01', '2023-01-02'],
        'Holiday': [True, False]
    }
    return pd.DataFrame(data)

def test_fetch_external_data():
    mock_response = {'key': 'value'}
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status = lambda: None
        result = fetch_external_data('http://example.com/api', params={'param': 'value'})
        assert result == mock_response
        mock_get.assert_called_once_with('http://example.com/api', params={'param': 'value'})

def test_merge_datasets(sample_dataframe, external_dataframe):
    merged_df = merge_datasets(sample_dataframe, external_dataframe, on_column='Date', how='left')
    assert len(merged_df) == len(sample_dataframe)
    assert 'Holiday' in merged_df.columns
    assert merged_df['Holiday'].iloc[0] == True
    assert pd.isna(merged_df['Holiday'].iloc[2])

def test_calculate_moving_average(sample_dataframe):
    enriched_df = calculate_moving_average(sample_dataframe, column='Sales', window=2)
    assert 'Sales_ma_2' in enriched_df.columns
    assert pd.isna(enriched_df['Sales_ma_2'].iloc[0])
    assert enriched_df['Sales_ma_2'].iloc[1] == 125.0
    assert enriched_df['Sales_ma_2'].iloc[2] == 135.0
    assert enriched_df['Sales_ma_2'].iloc[3] == 150.0 