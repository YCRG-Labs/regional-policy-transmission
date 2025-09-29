"""
Computation optimizer for parallel processing and performance improvements.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import logging
from functools import partial
import warnings

from ..econometric.models import RegionalParameters, EstimationResults
from ..data.models import RegionalDataset

logger = logging.getLogger(__name__)

class ComputationOptimizer:
    """Optimizes computational performance through parallel processing and vectorization."""
    
    def __init__(self, max_workers: Optional[int] = None, use_gpu: bool = False):
        self.max_workers = max_workers or min(cpu_count(), 8)  # Limit to 8 cores max
        self.use_gpu = use_gpu and self._check_gpu_availability()
        
        logger.info(f"Initialized optimizer with {self.max_workers} workers, GPU: {self.use_gpu}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except (ImportError, Exception):
            return False
    
    def optimize_regional_estimation(self, 
                                   estimation_func: Callable,
                                   regional_data: RegionalDataset,
                                   regions: List[str],
                                   **kwargs) -> Dict[str, Any]:
        """Parallelize regional parameter estimation across regions."""
        
        logger.info(f"Starting parallel estimation for {len(regions)} regions")
        
        # Prepare data chunks for each region
        region_chunks = self._prepare_regional_chunks(regional_data, regions)
        
        # Create partial function with fixed parameters
        estimation_partial = partial(self._estimate_single_region, 
                                   estimation_func=estimation_func, **kwargs)
        
        results = {}
        failed_regions = []
        
        # Use ProcessPoolExecutor for CPU-intensive estimation
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_region = {
                executor.submit(estimation_partial, region, data): region 
                for region, data in region_chunks.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per region
                    results[region] = result
                    logger.debug(f"Completed estimation for region {region}")
                except Exception as e:
                    logger.error(f"Failed estimation for region {region}: {str(e)}")
                    failed_regions.append(region)
        
        if failed_regions:
            logger.warning(f"Failed to estimate parameters for regions: {failed_regions}")
        
        logger.info(f"Completed parallel estimation: {len(results)}/{len(regions)} successful")
        
        return {
            'results': results,
            'failed_regions': failed_regions,
            'success_rate': len(results) / len(regions)
        }
    
    def _prepare_regional_chunks(self, 
                               regional_data: RegionalDataset, 
                               regions: List[str]) -> Dict[str, Dict[str, Any]]:
        """Prepare data chunks for parallel processing."""
        chunks = {}
        
        for region in regions:
            try:
                region_data = regional_data.get_region_data(region)
                chunks[region] = {
                    'output_gap': region_data.get('output_gap', pd.Series()),
                    'inflation': region_data.get('inflation', pd.Series()),
                    'interest_rate': regional_data.interest_rates,
                    'metadata': regional_data.metadata
                }
            except Exception as e:
                logger.warning(f"Could not prepare data for region {region}: {str(e)}")
        
        return chunks
    
    def _estimate_single_region(self, 
                              region: str, 
                              region_data: Dict[str, Any],
                              estimation_func: Callable,
                              **kwargs) -> Dict[str, Any]:
        """Estimate parameters for a single region."""
        try:
            # Convert back to appropriate format for estimation
            result = estimation_func(region_data, **kwargs)
            return {
                'region': region,
                'parameters': result,
                'success': True
            }
        except Exception as e:
            return {
                'region': region,
                'error': str(e),
                'success': False
            }
    
    def optimize_matrix_operations(self, 
                                 operation: str,
                                 matrices: List[np.ndarray],
                                 **kwargs) -> np.ndarray:
        """Optimize matrix operations using vectorization and GPU if available."""
        
        if self.use_gpu:
            return self._gpu_matrix_operation(operation, matrices, **kwargs)
        else:
            return self._cpu_matrix_operation(operation, matrices, **kwargs)
    
    def _cpu_matrix_operation(self, 
                            operation: str, 
                            matrices: List[np.ndarray],
                            **kwargs) -> np.ndarray:
        """Optimized CPU matrix operations."""
        
        # Use BLAS-optimized operations where possible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if operation == 'spatial_weights':
                return self._compute_spatial_weights_vectorized(matrices, **kwargs)
            elif operation == 'gmm_moments':
                return self._compute_gmm_moments_vectorized(matrices, **kwargs)
            elif operation == 'covariance':
                return self._compute_covariance_optimized(matrices[0], **kwargs)
            elif operation == 'eigendecomposition':
                return self._compute_eigendecomposition_optimized(matrices[0], **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
    
    def _gpu_matrix_operation(self, 
                            operation: str, 
                            matrices: List[np.ndarray],
                            **kwargs) -> np.ndarray:
        """GPU-accelerated matrix operations."""
        try:
            import cupy as cp
            
            # Transfer to GPU
            gpu_matrices = [cp.asarray(m) for m in matrices]
            
            if operation == 'spatial_weights':
                result = self._compute_spatial_weights_gpu(gpu_matrices, **kwargs)
            elif operation == 'gmm_moments':
                result = self._compute_gmm_moments_gpu(gpu_matrices, **kwargs)
            elif operation == 'covariance':
                result = cp.cov(gpu_matrices[0].T)
            elif operation == 'eigendecomposition':
                result = cp.linalg.eigh(gpu_matrices[0])
            else:
                raise ValueError(f"Unknown GPU operation: {operation}")
            
            # Transfer back to CPU
            return cp.asnumpy(result)
            
        except Exception as e:
            logger.warning(f"GPU operation failed, falling back to CPU: {str(e)}")
            return self._cpu_matrix_operation(operation, matrices, **kwargs)
    
    def _compute_spatial_weights_vectorized(self, 
                                          matrices: List[np.ndarray],
                                          weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)) -> np.ndarray:
        """Vectorized spatial weights computation."""
        
        if len(matrices) != 4:
            raise ValueError("Expected 4 matrices: trade, migration, financial, distance")
        
        trade_matrix, migration_matrix, financial_matrix, distance_matrix = matrices
        w_trade, w_migration, w_financial, w_distance = weights
        
        # Normalize each matrix
        normalized_matrices = []
        for matrix in matrices:
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            normalized_matrices.append(matrix / row_sums)
        
        # Weighted combination
        spatial_weights = (w_trade * normalized_matrices[0] + 
                          w_migration * normalized_matrices[1] + 
                          w_financial * normalized_matrices[2] + 
                          w_distance * normalized_matrices[3])
        
        return spatial_weights
    
    def _compute_gmm_moments_vectorized(self, 
                                      matrices: List[np.ndarray],
                                      **kwargs) -> np.ndarray:
        """Vectorized GMM moment computation."""
        
        if len(matrices) < 2:
            raise ValueError("Expected at least 2 matrices for GMM moments")
        
        residuals, instruments = matrices[0], matrices[1]
        
        # Compute moments: E[g(θ)] = E[Z'u]
        moments = np.dot(instruments.T, residuals) / residuals.shape[0]
        
        return moments
    
    def _compute_covariance_optimized(self, 
                                    data: np.ndarray,
                                    method: str = 'sample') -> np.ndarray:
        """Optimized covariance computation."""
        
        if method == 'sample':
            return np.cov(data.T)
        elif method == 'robust':
            # Robust covariance using Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            return lw.fit(data).covariance_
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    
    def _compute_eigendecomposition_optimized(self, 
                                            matrix: np.ndarray,
                                            method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """Optimized eigendecomposition."""
        
        if method == 'auto':
            # Choose method based on matrix size
            if matrix.shape[0] < 1000:
                return np.linalg.eigh(matrix)
            else:
                # Use sparse methods for large matrices
                from scipy.sparse.linalg import eigsh
                return eigsh(matrix, k=min(10, matrix.shape[0]-1))
        else:
            return np.linalg.eigh(matrix)
    
    def optimize_data_loading(self, 
                            data_loader: Callable,
                            data_sources: List[str],
                            **kwargs) -> Dict[str, Any]:
        """Parallelize data loading operations."""
        
        logger.info(f"Starting parallel data loading for {len(data_sources)} sources")
        
        results = {}
        failed_sources = []
        
        # Use ThreadPoolExecutor for I/O-bound data loading
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(data_sources))) as executor:
            # Submit all tasks
            future_to_source = {
                executor.submit(data_loader, source, **kwargs): source 
                for source in data_sources
            }
            
            # Collect results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result(timeout=60)  # 1 minute timeout per source
                    results[source] = result
                    logger.debug(f"Loaded data from source {source}")
                except Exception as e:
                    logger.error(f"Failed to load data from source {source}: {str(e)}")
                    failed_sources.append(source)
        
        logger.info(f"Completed parallel data loading: {len(results)}/{len(data_sources)} successful")
        
        return {
            'results': results,
            'failed_sources': failed_sources,
            'success_rate': len(results) / len(data_sources)
        }
    
    def get_optimization_recommendations(self, 
                                       profiling_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        
        recommendations = []
        
        # Check for parallelization opportunities
        if 'slowest_functions' in profiling_results:
            for func_name, exec_time in profiling_results['slowest_functions']:
                if 'estimation' in func_name.lower() and exec_time > 5.0:
                    recommendations.append(f"Consider parallelizing {func_name} - takes {exec_time:.1f}s")
                
                if 'matrix' in func_name.lower() and exec_time > 2.0:
                    recommendations.append(f"Consider GPU acceleration for {func_name}")
        
        # Check memory usage
        if 'memory_intensive_functions' in profiling_results:
            for func_name, memory in profiling_results['memory_intensive_functions']:
                if memory > 500:  # More than 500MB
                    recommendations.append(f"Consider memory optimization for {func_name} - uses {memory:.1f}MB")
        
        # General recommendations
        if self.use_gpu:
            recommendations.append("GPU acceleration is available - ensure matrix operations use GPU")
        else:
            recommendations.append("Consider installing CuPy for GPU acceleration")
        
        recommendations.append(f"Parallel processing configured for {self.max_workers} workers")
        
        return recommendations