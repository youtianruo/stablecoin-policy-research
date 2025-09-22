"""
Tests for I/O utilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.io import save_data, load_data, ensure_directory, get_file_info


class TestIO:
    """Test I/O utilities."""
    
    def test_save_load_dataframe_parquet(self):
        """Test saving and loading DataFrame as parquet."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        test_data.index = pd.date_range('2020-01-01', periods=5, freq='D')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save data
            save_data(test_data, 'test_data', temp_dir, file_format='parquet')
            
            # Load data
            loaded_data = load_data('test_data', temp_dir, file_format='parquet')
            
            # Check if data is preserved
            pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_save_load_dataframe_csv(self):
        """Test saving and loading DataFrame as CSV."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        test_data.index = pd.date_range('2020-01-01', periods=5, freq='D')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save data
            save_data(test_data, 'test_data', temp_dir, file_format='csv')
            
            # Load data
            loaded_data = load_data('test_data', temp_dir, file_format='csv')
            
            # Check if data is preserved (within tolerance for CSV)
            pd.testing.assert_frame_equal(test_data, loaded_data, check_exact=False)
    
    def test_save_load_pickle(self):
        """Test saving and loading Python objects as pickle."""
        # Create test data
        test_data = {
            'list': [1, 2, 3, 4, 5],
            'dict': {'a': 1, 'b': 2, 'c': 3},
            'numpy_array': np.array([1, 2, 3, 4, 5])
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save data
            save_data(test_data, 'test_data', temp_dir, file_format='pickle')
            
            # Load data
            loaded_data = load_data('test_data', temp_dir, file_format='pickle')
            
            # Check if data is preserved
            assert loaded_data == test_data
    
    def test_save_load_json(self):
        """Test saving and loading data as JSON."""
        # Create test data
        test_data = {
            'list': [1, 2, 3, 4, 5],
            'dict': {'a': 1, 'b': 2, 'c': 3},
            'string': 'test'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save data
            save_data(test_data, 'test_data', temp_dir, file_format='json')
            
            # Load data
            loaded_data = load_data('test_data', temp_dir, file_format='json')
            
            # Check if data is preserved
            assert loaded_data == test_data
    
    def test_auto_detect_format(self):
        """Test automatic format detection."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save as parquet
            save_data(test_data, 'test_data', temp_dir, file_format='parquet')
            
            # Load without specifying format
            loaded_data = load_data('test_data', temp_dir)
            
            # Check if data is preserved
            pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested directory
            nested_dir = os.path.join(temp_dir, 'level1', 'level2', 'level3')
            ensure_directory(nested_dir)
            
            # Check if directory exists
            assert os.path.exists(nested_dir)
            assert os.path.isdir(nested_dir)
    
    def test_get_file_info(self):
        """Test file information retrieval."""
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save data
            file_path = os.path.join(temp_dir, 'test_data.parquet')
            test_data.to_parquet(file_path)
            
            # Get file info
            info = get_file_info(file_path)
            
            # Check basic info
            assert info['filename'] == 'test_data.parquet'
            assert info['extension'] == '.parquet'
            assert info['size_bytes'] > 0
            assert info['size_mb'] > 0
            
            # Check DataFrame-specific info
            assert info['shape'] == (5, 2)
            assert info['columns'] == ['A', 'B']
            assert info['index_type'] == 'RangeIndex'
    
    def test_file_not_found_error(self):
        """Test error handling for missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                load_data('nonexistent_file', temp_dir)
    
    def test_invalid_format_error(self):
        """Test error handling for invalid formats."""
        test_data = pd.DataFrame({'A': [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                save_data(test_data, 'test_data', temp_dir, file_format='invalid_format')
    
    def test_list_data_files(self):
        """Test listing data files."""
        from utils.io import list_data_files
        
        # Create test files
        test_data = pd.DataFrame({'A': [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save multiple files
            save_data(test_data, 'file1', temp_dir, file_format='parquet')
            save_data(test_data, 'file2', temp_dir, file_format='csv')
            save_data(test_data, 'file3', temp_dir, file_format='parquet')
            
            # List all files
            all_files = list_data_files(temp_dir)
            assert len(all_files) == 3
            
            # List parquet files only
            parquet_files = list_data_files(temp_dir, '*.parquet')
            assert len(parquet_files) == 2
            
            # List csv files only
            csv_files = list_data_files(temp_dir, '*.csv')
            assert len(csv_files) == 1
