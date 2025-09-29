"""
Memory manager for efficient handling of large datasets.
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Iterator
from dataclasses import dataclass
import logging
import warnings
from contextlib import contextmanager
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Container for memory usage statistics."""
    total_memory: float  # GB
    available_memory: float  # GB
    used_memory: float  # GB
    memory_percent: float
    process_memory: float  # GB

class MemoryManager:
    """Manages memory usage for large datasets and computations."""
    
    def __init__(self, 
                 memory_limit_gb: Optional[float] = None,
                 enable_disk_cache: bool = True,
                 temp_dir: Optional[str] = None):
        
        self.memory_limit_gb = memory_limit_gb or self._get_safe_memory_limit()
        self.enable_disk_cache = enable_disk_cache
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.cached_arrays: Dict[str, str] = {}  # Maps cache keys to file paths
        self.memory_pool: Dict[str, np.ndarray] = {}
        
        logger.info(f"Memory manager initialized with {self.memory_limit_gb:.1f}GB limit")
    
    def _get_safe_memory_limit(self) -> float:
        """Get safe memory limit (80% of available memory)."""
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        return total_memory * 0.8
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        vm = psutil.virtual_memory()
        process = psutil.Process()
        
        return MemoryStats(
            total_memory=vm.total / (1024**3),
            available_memory=vm.available / (1024**3),
            used_memory=vm.used / (1024**3),
            memory_percent=vm.percent,
            process_memory=process.memory_info().rss / (1024**3)
        )
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        stats = self.get_memory_stats()
        return stats.process_memory < self.memory_limit_gb
    
    @contextmanager
    def memory_context(self, operation_name: str):
        """Context manager for monitoring memory usage during operations."""
        start_stats = self.get_memory_stats()
        logger.debug(f"Starting {operation_name} - Memory: {start_stats.process_memory:.2f}GB")
        
        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            memory_delta = end_stats.process_memory - start_stats.process_memory
            logger.debug(f"Completed {operation_name} - Memory delta: {memory_delta:+.2f}GB")
            
            if memory_delta > 1.0:  # More than 1GB increase
                logger.warning(f"{operation_name} used {memory_delta:.2f}GB memory")
                self._suggest_optimization(operation_name, memory_delta)
    
    def _suggest_optimization(self, operation_name: str, memory_usage: float):
        """Suggest memory optimization strategies."""
        suggestions = []
        
        if memory_usage > 2.0:
            suggestions.append("Consider chunking data processing")
            suggestions.append("Enable disk caching for intermediate results")
        
        if memory_usage > 5.0:
            suggestions.append("Use memory-mapped arrays for large datasets")
            suggestions.append("Consider distributed processing")
        
        for suggestion in suggestions:
            logger.info(f"Memory optimization suggestion for {operation_name}: {suggestion}")
    
    def optimize_dataframe(self, 
                          df: pd.DataFrame, 
                          categorical_threshold: int = 50) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        
        original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert to categorical if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < categorical_threshold:
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame memory optimized: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                   f"({reduction:.1f}% reduction)")
        
        return df
    
    def create_memory_mapped_array(self, 
                                 shape: tuple, 
                                 dtype: np.dtype = np.float64,
                                 cache_key: Optional[str] = None) -> np.ndarray:
        """Create memory-mapped array for large datasets."""
        
        if cache_key and cache_key in self.cached_arrays:
            filepath = self.cached_arrays[cache_key]
            if os.path.exists(filepath):
                return np.memmap(filepath, dtype=dtype, mode='r+', shape=shape)
        
        # Create new memory-mapped file
        filepath = os.path.join(self.temp_dir, f"memmap_{id(self)}_{len(self.cached_arrays)}.dat")
        
        try:
            mmap_array = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)
            
            if cache_key:
                self.cached_arrays[cache_key] = filepath
            
            logger.debug(f"Created memory-mapped array: {shape} at {filepath}")
            return mmap_array
            
        except Exception as e:
            logger.error(f"Failed to create memory-mapped array: {str(e)}")
            # Fall back to regular array
            return np.zeros(shape, dtype=dtype)
    
    def chunk_array_processing(self, 
                             array: np.ndarray,
                             processing_func: callable,
                             chunk_size: Optional[int] = None,
                             axis: int = 0) -> np.ndarray:
        """Process large arrays in chunks to manage memory."""
        
        if chunk_size is None:
            # Estimate chunk size based on available memory
            element_size = array.dtype.itemsize
            available_memory_bytes = self.memory_limit_gb * (1024**3) * 0.5  # Use 50% of limit
            chunk_size = max(1, int(available_memory_bytes / (element_size * np.prod(array.shape[1:]))))
        
        logger.debug(f"Processing array in chunks of size {chunk_size}")
        
        results = []
        total_chunks = (array.shape[axis] + chunk_size - 1) // chunk_size
        
        for i in range(0, array.shape[axis], chunk_size):
            end_idx = min(i + chunk_size, array.shape[axis])
            
            if axis == 0:
                chunk = array[i:end_idx]
            elif axis == 1:
                chunk = array[:, i:end_idx]
            else:
                raise ValueError("Only axis 0 and 1 supported for chunking")
            
            # Process chunk
            with self.memory_context(f"chunk_{i//chunk_size + 1}/{total_chunks}"):
                result = processing_func(chunk)
                results.append(result)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        # Combine results
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=axis)
        else:
            return results
    
    def efficient_matrix_multiply(self, 
                                A: np.ndarray, 
                                B: np.ndarray,
                                chunk_size: Optional[int] = None) -> np.ndarray:
        """Efficient matrix multiplication for large matrices."""
        
        # Check if matrices fit in memory
        result_size_gb = (A.shape[0] * B.shape[1] * 8) / (1024**3)  # Assuming float64
        
        if result_size_gb > self.memory_limit_gb * 0.8:
            logger.info(f"Large matrix multiplication ({result_size_gb:.1f}GB) - using chunked approach")
            return self._chunked_matrix_multiply(A, B, chunk_size)
        else:
            return np.dot(A, B)
    
    def _chunked_matrix_multiply(self, 
                               A: np.ndarray, 
                               B: np.ndarray,
                               chunk_size: Optional[int] = None) -> np.ndarray:
        """Chunked matrix multiplication for memory efficiency."""
        
        if chunk_size is None:
            # Estimate chunk size
            available_memory_bytes = self.memory_limit_gb * (1024**3) * 0.3  # Use 30% of limit
            chunk_size = max(1, int(available_memory_bytes / (B.shape[1] * 8)))
        
        result = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        
        for i in range(0, A.shape[0], chunk_size):
            end_idx = min(i + chunk_size, A.shape[0])
            chunk_result = np.dot(A[i:end_idx], B)
            result[i:end_idx] = chunk_result
            
            # Clear intermediate results
            del chunk_result
            gc.collect()
        
        return result
    
    def create_data_iterator(self, 
                           data: Union[pd.DataFrame, np.ndarray],
                           batch_size: int) -> Iterator[Union[pd.DataFrame, np.ndarray]]:
        """Create memory-efficient iterator for large datasets."""
        
        total_size = len(data)
        
        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            
            if isinstance(data, pd.DataFrame):
                yield data.iloc[start_idx:end_idx].copy()
            else:
                yield data[start_idx:end_idx].copy()
    
    def cleanup_cache(self):
        """Clean up cached memory-mapped files."""
        
        for cache_key, filepath in self.cached_arrays.items():
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed cached file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove cached file {filepath}: {str(e)}")
        
        self.cached_arrays.clear()
        self.memory_pool.clear()
        gc.collect()
        
        logger.info("Memory cache cleaned up")
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats.memory_percent > 80:
            recommendations.append("System memory usage is high - consider reducing dataset size")
        
        if stats.process_memory > self.memory_limit_gb * 0.8:
            recommendations.append("Process approaching memory limit - enable disk caching")
        
        if len(self.cached_arrays) > 10:
            recommendations.append("Many cached arrays - consider cleanup")
        
        if not self.enable_disk_cache:
            recommendations.append("Enable disk caching for better memory management")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when memory manager is destroyed."""
        try:
            self.cleanup_cache()
        except Exception:
            pass  # Ignore cleanup errors during destruction