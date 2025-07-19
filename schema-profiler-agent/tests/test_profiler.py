import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the schema_profiler module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_profiler.profiler import Profiler, profile_dataframe, get_semantic_profile
from schema_profiler.yaml_writer import write_profile_yaml, write_schema_yaml, write_yaml_files
import tempfile
import yaml

class TestProfiler:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, None, 40],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'IT', 'HR', 'Finance'],
            'start_date': pd.to_datetime(['2020-01-01', '2019-05-15', '2021-03-10', '2018-11-20', '2022-07-05'])
        })
    
    def test_profiler_initialization(self, sample_dataframe):
        """Test that Profiler initializes correctly"""
        profiler = Profiler(sample_dataframe)
        assert profiler.df is not None
        assert len(profiler.df) == 5
    
    def test_get_row_count(self, sample_dataframe):
        """Test row count calculation"""
        profiler = Profiler(sample_dataframe)
        assert profiler.get_row_count() == 5
    
    def test_get_null_percentages(self, sample_dataframe):
        """Test null percentage calculation"""
        profiler = Profiler(sample_dataframe)
        null_percentages = profiler.get_null_percentages()
        
        assert null_percentages['age'] == 20.0  # 1 null out of 5
        assert null_percentages['name'] == 0.0  # No nulls
        assert null_percentages['salary'] == 0.0  # No nulls
    
    def test_get_cardinality(self, sample_dataframe):
        """Test cardinality calculation"""
        profiler = Profiler(sample_dataframe)
        cardinality = profiler.get_cardinality()
        
        assert cardinality['id'] == 5  # All unique
        assert cardinality['department'] == 3  # HR, IT, Finance
        assert cardinality['name'] == 5  # All unique names
    
    def test_get_top_values(self, sample_dataframe):
        """Test top values extraction"""
        profiler = Profiler(sample_dataframe)
        top_values = profiler.get_top_values(top_n=3)
        
        # Department should have IT appearing twice
        assert 'IT' in top_values['department']
        assert 'HR' in top_values['department']
        assert 'Finance' in top_values['department']
        
        # All names should be unique, so all should appear in top values
        assert len(top_values['name']) == 5
    
    def test_profile_dataframe_function(self, sample_dataframe):
        """Test the standalone profile_dataframe function"""
        profile = profile_dataframe(sample_dataframe)
        
        assert 'row_count' in profile
        assert 'null_percentage' in profile
        assert 'cardinality' in profile
        assert 'top_values' in profile
        
        assert profile['row_count'] == 5
        assert profile['null_percentage']['age'] == 20.0
    
    def test_profile_dataframe_complete(self, sample_dataframe):
        """Test complete profiling of a DataFrame"""
        profiler = Profiler(sample_dataframe)
        profile = profiler.profile_dataframe()
        
        # Check all required keys are present
        required_keys = ['row_count', 'null_percentage', 'cardinality', 'top_values']
        for key in required_keys:
            assert key in profile
        
        # Check data types
        assert isinstance(profile['row_count'], int)
        assert isinstance(profile['null_percentage'], dict)
        assert isinstance(profile['cardinality'], dict)
        assert isinstance(profile['top_values'], dict)

class TestSemanticProfiling:
    
    @patch('schema_profiler.profiler.litellm.completion')
    def test_get_semantic_profile(self, mock_completion):
        """Test semantic profiling with mocked LLM response"""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
dataset: test_data
primary_key: id
dimensions:
  - name: department
    semantic_type: category
  - name: start_date
    semantic_type: date
measures:
  - name: salary
    data_type: int
    aggregation: sum
  - name: age
    data_type: int
    aggregation: avg
"""
        mock_completion.return_value = mock_response
        
        sample_profile = {
            'row_count': 5,
            'null_percentage': {'age': 20.0, 'salary': 0.0},
            'cardinality': {'id': 5, 'department': 3},
            'top_values': {'department': ['IT', 'HR', 'Finance']}
        }
        
        result = get_semantic_profile(sample_profile)
        
        # Check that litellm was called
        mock_completion.assert_called_once()
        
        # Check that the result contains expected YAML structure
        assert 'dataset: test_data' in result
        assert 'primary_key: id' in result
        assert 'dimensions:' in result
        assert 'measures:' in result

class TestYamlWriter:
    
    def test_write_profile_yaml(self):
        """Test writing profile YAML file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            sample_profile = {
                'row_count': 100,
                'null_percentage': {'col1': 5.0, 'col2': 0.0},
                'cardinality': {'col1': 95, 'col2': 10},
                'top_values': {'col1': ['A', 'B', 'C'], 'col2': ['X', 'Y']}
            }
            
            write_profile_yaml(temp_path, 'test_dataset', sample_profile)
            
            # Check file was created
            profile_file = temp_path / 'test_dataset.profile.yaml'
            assert profile_file.exists()
            
            # Check file contents
            with open(profile_file, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data['row_count'] == 100
            assert loaded_data['null_percentage']['col1'] == 5.0
    
    def test_write_schema_yaml(self):
        """Test writing schema YAML file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            sample_schema = """
dataset: test_data
primary_key: id
dimensions:
  - name: category
    semantic_type: category
measures:
  - name: value
    data_type: int
    aggregation: sum
"""
            
            write_schema_yaml(temp_path, 'test_dataset', sample_schema)
            
            # Check file was created
            schema_file = temp_path / 'test_dataset.schema.yaml'
            assert schema_file.exists()
            
            # Check file contents
            with open(schema_file, 'r') as f:
                content = f.read()
            
            assert 'dataset: test_data' in content
            assert 'primary_key: id' in content
    
    def test_write_yaml_files(self):
        """Test writing both profile and schema YAML files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            sample_profile = {'row_count': 50}
            sample_schema = 'dataset: test\nprimary_key: id'
            
            write_yaml_files(temp_path, 'combined_test', sample_profile, sample_schema)
            
            # Check both files were created
            profile_file = temp_path / 'combined_test.profile.yaml'
            schema_file = temp_path / 'combined_test.schema.yaml'
            
            assert profile_file.exists()
            assert schema_file.exists()

class TestErrorHandling:
    
    def test_empty_dataframe(self):
        """Test profiling an empty DataFrame"""
        empty_df = pd.DataFrame()
        profiler = Profiler(empty_df)
        
        assert profiler.get_row_count() == 0
        
        # Null percentages should be empty dict for empty DataFrame
        null_percentages = profiler.get_null_percentages()
        assert isinstance(null_percentages, dict)
    
    def test_dataframe_with_all_nulls(self):
        """Test profiling a DataFrame with all null values"""
        null_df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [np.nan, np.nan, np.nan]
        })
        profiler = Profiler(null_df)
        
        null_percentages = profiler.get_null_percentages()
        assert null_percentages['col1'] == 100.0
        assert null_percentages['col2'] == 100.0
    
    def test_single_row_dataframe(self):
        """Test profiling a DataFrame with a single row"""
        single_row_df = pd.DataFrame({
            'id': [1],
            'name': ['Alice'],
            'value': [100]
        })
        profiler = Profiler(single_row_df)
        
        profile = profiler.profile_dataframe()
        assert profile['row_count'] == 1
        assert profile['cardinality']['id'] == 1
        assert profile['null_percentage']['name'] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
