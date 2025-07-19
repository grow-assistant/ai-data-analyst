import pytest
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup test environment
from tests.setup_test_env import setup_test_environment
setup_test_environment()

def test_tdsx_loading_function():
    """Test the TDSX loading function directly"""
    
    # Import the function after setting up the path
    # Add the data-loader-agent directory to path
    data_loader_path = PROJECT_ROOT / 'data-loader-agent'
    sys.path.insert(0, str(data_loader_path))
    
    from data_loader.agent import load_tdsx_data
    
    # Test with the AI_DS.tdsx file
    tdsx_path = "AI_DS.tdsx"  # Should be found in data directory
    
    result = load_tdsx_data(tdsx_path)
    
    print(f"TDSX Loading Result:\n{result}")
    
    # Check that the function returned useful information
    assert "Successfully extracted TDSX file" in result
    assert "AI_DS.tdsx" in result
    assert "File size:" in result
    
    # Check for either TDS or Hyper file information
    assert "Found TDS files:" in result or "Found Hyper files:" in result
    
    print("✅ TDSX loading function test passed!")

def test_tdsx_loading_with_full_path():
    """Test TDSX loading with full path"""
    
    from data_loader.agent import load_tdsx_data
    
    # Use full path
    tdsx_path = PROJECT_ROOT / 'data-loader-agent' / 'data' / 'AI_DS.tdsx'
    
    result = load_tdsx_data(str(tdsx_path))
    
    print(f"Full Path TDSX Loading Result:\n{result}")
    
    # Should work with full path too
    assert "Successfully extracted TDSX file" in result
    print("✅ Full path TDSX loading test passed!")

def test_tdsx_loading_nonexistent_file():
    """Test TDSX loading with non-existent file"""
    
    from data_loader.agent import load_tdsx_data
    
    result = load_tdsx_data("nonexistent.tdsx")
    
    print(f"Non-existent file result: {result}")
    
    # Should handle missing files gracefully
    assert "not found" in result
    print("✅ Non-existent file handling test passed!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 