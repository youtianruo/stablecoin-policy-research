"""
Input/output utilities for data management.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Any
import pickle
import json


def load_data(
    filename: str, 
    data_dir: str, 
    file_format: Optional[str] = None
) -> Union[pd.DataFrame, Any]:
    """
    Load data from various file formats.
    
    Args:
        filename: Name of the file (without extension)
        data_dir: Directory containing the data files
        file_format: File format ('parquet', 'csv', 'pickle', 'json'). 
                    If None, will try to detect automatically.
        
    Returns:
        Loaded data (DataFrame or other object)
    """
    data_path = Path(data_dir)
    
    # Try to detect file format if not specified
    if file_format is None:
        for fmt in ['parquet', 'csv', 'pickle', 'json']:
            file_path = data_path / f"{filename}.{fmt}"
            if file_path.exists():
                file_format = fmt
                break
        
        if file_format is None:
            raise FileNotFoundError(f"No data file found for {filename} in {data_dir}")
    
    file_path = data_path / f"{filename}.{file_format}"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load based on format
    if file_format == 'parquet':
        return pd.read_parquet(file_path)
    elif file_format == 'csv':
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif file_format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_format == 'json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def save_data(
    data: Any, 
    filename: str, 
    data_dir: str, 
    file_format: str = 'parquet',
    **kwargs
) -> None:
    """
    Save data to various file formats.
    
    Args:
        data: Data to save (DataFrame or other object)
        filename: Name of the file (without extension)
        data_dir: Directory to save the data files
        file_format: File format ('parquet', 'csv', 'pickle', 'json')
        **kwargs: Additional arguments for the save function
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    file_path = data_path / f"{filename}.{file_format}"
    
    # Save based on format
    if file_format == 'parquet':
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, **kwargs)
        else:
            raise ValueError("Data must be a DataFrame for parquet format")
    
    elif file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, **kwargs)
        else:
            raise ValueError("Data must be a DataFrame for CSV format")
    
    elif file_format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    
    elif file_format == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, **kwargs)
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def ensure_directory(path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def list_data_files(data_dir: str, pattern: str = "*") -> list:
    """
    List data files in a directory.
    
    Args:
        data_dir: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    return list(data_path.glob(pattern))


def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get information about a data file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = path.stat()
    
    info = {
        'filename': path.name,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': pd.Timestamp.fromtimestamp(stat.st_mtime),
        'extension': path.suffix
    }
    
    # Add DataFrame info if it's a data file
    if path.suffix in ['.parquet', '.csv']:
        try:
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            info.update({
                'shape': df.shape,
                'columns': list(df.columns),
                'index_type': type(df.index).__name__,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            })
        except Exception:
            pass
    
    return info
