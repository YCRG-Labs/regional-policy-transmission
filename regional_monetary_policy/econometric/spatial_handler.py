"""
Spatial modeling infrastructure for regional monetary policy analysis.

This module implements the SpatialModelHandler class and related functionality
for constructing, validating, and working with spatial weight matrices in the
context of regional monetary policy transmission analysis.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
# Moran's I will be implemented manually
import logging

from ..exceptions import SpatialModelError


logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results from spatial weight matrix validation."""
    
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    properties: Dict[str, Any]
    
    def summary(self) -> str:
        """Generate validation summary report."""
        status = "VALID" if self.is_valid else "INVALID"
        
        report = f"Spatial Weight Matrix Validation: {status}\n"
        report += "=" * 50 + "\n"
        
        if self.properties:
            report += "Matrix Properties:\n"
            for prop, value in self.properties.items():
                report += f"  {prop}: {value}\n"
        
        if self.warnings:
            report += "\nWarnings:\n"
            for warning in self.warnings:
                report += f"  - {warning}\n"
        
        if self.errors:
            report += "\nErrors:\n"
            for error in self.errors:
                report += f"  - {error}\n"
        
        return report


@dataclass
class SpatialWeightResults:
    """Results from spatial weight matrix construction."""
    
    weight_matrix: np.ndarray
    component_matrices: Dict[str, np.ndarray]
    component_weights: Dict[str, float]
    validation_report: ValidationReport
    construction_method: str
    
    def get_eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of the spatial weight matrix."""
        return np.linalg.eigvals(self.weight_matrix)
    
    def get_spectral_radius(self) -> float:
        """Compute spectral radius (largest absolute eigenvalue)."""
        eigenvals = self.get_eigenvalues()
        return np.max(np.abs(eigenvals))


class SpatialModelHandler:
    """
    Handles construction and validation of spatial weight matrices for regional analysis.
    
    This class implements methods for combining trade, migration, financial, and distance
    data into spatial weight matrices, along with validation and diagnostic functionality.
    """
    
    def __init__(self, regions: List[str]):
        """
        Initialize the spatial model handler.
        
        Args:
            regions: List of region identifiers
        """
        self.regions = regions
        self.n_regions = len(regions)
        self.region_index = {region: i for i, region in enumerate(regions)}
        
        if self.n_regions < 2:
            raise SpatialModelError("At least 2 regions required for spatial modeling")
        
        logger.info(f"Initialized SpatialModelHandler for {self.n_regions} regions")
    
    def construct_weights(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        migration_data: Optional[pd.DataFrame] = None,
        financial_data: Optional[pd.DataFrame] = None,
        distance_matrix: Optional[np.ndarray] = None,
        weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        normalize_method: str = "row"
    ) -> SpatialWeightResults:
        """
        Construct spatial weight matrix from multiple data sources.
        
        Args:
            trade_data: Trade flow data between regions
            migration_data: Migration flow data between regions
            financial_data: Financial linkage data between regions
            distance_matrix: Geographic distance matrix
            weights: Weights for (trade, migration, financial, distance) components
            normalize_method: Normalization method ('row', 'spectral', 'none')
            
        Returns:
            SpatialWeightResults containing the constructed matrix and diagnostics
        """
        if len(weights) != 4:
            raise SpatialModelError("Must provide exactly 4 weights for components")
        
        if not np.isclose(sum(weights), 1.0):
            raise SpatialModelError("Component weights must sum to 1.0")
        
        logger.info("Constructing spatial weight matrix from multiple components")
        
        # Initialize component matrices
        component_matrices = {}
        
        # Trade component
        if trade_data is not None and weights[0] > 0:
            component_matrices['trade'] = self._process_trade_data(trade_data)
        else:
            component_matrices['trade'] = np.zeros((self.n_regions, self.n_regions))
        
        # Migration component
        if migration_data is not None and weights[1] > 0:
            component_matrices['migration'] = self._process_migration_data(migration_data)
        else:
            component_matrices['migration'] = np.zeros((self.n_regions, self.n_regions))
        
        # Financial component
        if financial_data is not None and weights[2] > 0:
            component_matrices['financial'] = self._process_financial_data(financial_data)
        else:
            component_matrices['financial'] = np.zeros((self.n_regions, self.n_regions))
        
        # Distance component
        if distance_matrix is not None and weights[3] > 0:
            component_matrices['distance'] = self._process_distance_matrix(distance_matrix)
        else:
            component_matrices['distance'] = np.zeros((self.n_regions, self.n_regions))
        
        # Combine components with weights
        W = (weights[0] * component_matrices['trade'] +
             weights[1] * component_matrices['migration'] +
             weights[2] * component_matrices['financial'] +
             weights[3] * component_matrices['distance'])
        
        # Normalize the matrix
        W_normalized = self._normalize_matrix(W, method=normalize_method)
        
        # Validate the constructed matrix
        validation_report = self.validate_spatial_matrix(W_normalized)
        
        component_weights = {
            'trade': weights[0],
            'migration': weights[1], 
            'financial': weights[2],
            'distance': weights[3]
        }
        
        return SpatialWeightResults(
            weight_matrix=W_normalized,
            component_matrices=component_matrices,
            component_weights=component_weights,
            validation_report=validation_report,
            construction_method=f"combined_{normalize_method}_normalized"
        )
    
    def _process_trade_data(self, trade_data: pd.DataFrame) -> np.ndarray:
        """Process trade flow data into spatial weight matrix component."""
        logger.debug("Processing trade data component")
        
        # Initialize trade matrix
        trade_matrix = np.zeros((self.n_regions, self.n_regions))
        
        # Expected columns: origin, destination, trade_flow
        required_cols = ['origin', 'destination', 'trade_flow']
        if not all(col in trade_data.columns for col in required_cols):
            raise SpatialModelError(f"Trade data must contain columns: {required_cols}")
        
        for _, row in trade_data.iterrows():
            origin = row['origin']
            destination = row['destination']
            flow = row['trade_flow']
            
            if origin in self.region_index and destination in self.region_index:
                i = self.region_index[origin]
                j = self.region_index[destination]
                trade_matrix[i, j] = flow
        
        # Normalize by row sums to get trade shares
        row_sums = np.sum(trade_matrix, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trade_matrix = trade_matrix / row_sums[:, np.newaxis]
        
        return trade_matrix
    
    def _process_migration_data(self, migration_data: pd.DataFrame) -> np.ndarray:
        """Process migration flow data into spatial weight matrix component."""
        logger.debug("Processing migration data component")
        
        # Initialize migration matrix
        migration_matrix = np.zeros((self.n_regions, self.n_regions))
        
        # Expected columns: origin, destination, migration_flow
        required_cols = ['origin', 'destination', 'migration_flow']
        if not all(col in migration_data.columns for col in required_cols):
            raise SpatialModelError(f"Migration data must contain columns: {required_cols}")
        
        for _, row in migration_data.iterrows():
            origin = row['origin']
            destination = row['destination']
            flow = row['migration_flow']
            
            if origin in self.region_index and destination in self.region_index:
                i = self.region_index[origin]
                j = self.region_index[destination]
                migration_matrix[i, j] = flow
        
        # Normalize by row sums to get migration shares
        row_sums = np.sum(migration_matrix, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        migration_matrix = migration_matrix / row_sums[:, np.newaxis]
        
        return migration_matrix
    
    def _process_financial_data(self, financial_data: pd.DataFrame) -> np.ndarray:
        """Process financial linkage data into spatial weight matrix component."""
        logger.debug("Processing financial data component")
        
        # Initialize financial matrix
        financial_matrix = np.zeros((self.n_regions, self.n_regions))
        
        # Expected columns: region1, region2, financial_linkage
        required_cols = ['region1', 'region2', 'financial_linkage']
        if not all(col in financial_data.columns for col in required_cols):
            raise SpatialModelError(f"Financial data must contain columns: {required_cols}")
        
        for _, row in financial_data.iterrows():
            region1 = row['region1']
            region2 = row['region2']
            linkage = row['financial_linkage']
            
            if region1 in self.region_index and region2 in self.region_index:
                i = self.region_index[region1]
                j = self.region_index[region2]
                # Financial linkages are typically symmetric
                financial_matrix[i, j] = linkage
                financial_matrix[j, i] = linkage
        
        # For financial data, we normalize while preserving symmetry
        # Use the maximum of row and column sums for each element
        row_sums = np.sum(financial_matrix, axis=1)
        col_sums = np.sum(financial_matrix, axis=0)
        
        # Create symmetric normalization factors
        norm_factors = np.maximum(row_sums, col_sums)
        norm_factors[norm_factors == 0] = 1  # Avoid division by zero
        
        # Apply symmetric normalization
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if norm_factors[i] > 0 and norm_factors[j] > 0:
                    financial_matrix[i, j] = financial_matrix[i, j] / np.sqrt(norm_factors[i] * norm_factors[j])
        
        return financial_matrix
    
    def _process_distance_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Process geographic distance matrix into spatial weight matrix component."""
        logger.debug("Processing distance matrix component")
        
        if distance_matrix.shape != (self.n_regions, self.n_regions):
            raise SpatialModelError(
                f"Distance matrix shape {distance_matrix.shape} doesn't match "
                f"number of regions {self.n_regions}"
            )
        
        # Convert distances to weights using inverse distance
        # Add small constant to avoid division by zero for same-region distances
        distance_weights = 1.0 / (distance_matrix + 1e-6)
        
        # Set diagonal to zero (no self-interaction)
        np.fill_diagonal(distance_weights, 0)
        
        # Normalize by row sums
        row_sums = np.sum(distance_weights, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        distance_weights = distance_weights / row_sums[:, np.newaxis]
        
        return distance_weights
    
    def _normalize_matrix(self, matrix: np.ndarray, method: str = "row") -> np.ndarray:
        """
        Normalize spatial weight matrix.
        
        Args:
            matrix: Input matrix to normalize
            method: Normalization method ('row', 'spectral', 'none')
            
        Returns:
            Normalized matrix
        """
        if method == "none":
            return matrix.copy()
        
        elif method == "row":
            # Row-standardize: each row sums to 1
            row_sums = np.sum(matrix, axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return matrix / row_sums[:, np.newaxis]
        
        elif method == "spectral":
            # Spectral normalization: largest eigenvalue = 1
            eigenvals = np.linalg.eigvals(matrix)
            max_eigenval = np.max(np.real(eigenvals))
            if max_eigenval > 0:
                return matrix / max_eigenval
            else:
                return matrix
        
        else:
            raise SpatialModelError(f"Unknown normalization method: {method}")
    
    def validate_spatial_matrix(self, W: np.ndarray) -> ValidationReport:
        """
        Validate spatial weight matrix properties.
        
        Args:
            W: Spatial weight matrix to validate
            
        Returns:
            ValidationReport with validation results
        """
        logger.debug("Validating spatial weight matrix")
        
        errors = []
        warnings = []
        properties = {}
        
        # Check basic properties
        if W.shape != (self.n_regions, self.n_regions):
            errors.append(f"Matrix shape {W.shape} doesn't match regions {self.n_regions}")
        
        # Check for non-negative entries
        if np.any(W < 0):
            warnings.append("Matrix contains negative entries")
        
        # Check diagonal elements
        diagonal_sum = np.sum(np.diag(W))
        properties['diagonal_sum'] = diagonal_sum
        if diagonal_sum > 1e-10:
            warnings.append(f"Non-zero diagonal elements (sum = {diagonal_sum:.6f})")
        
        # Check row sums
        row_sums = np.sum(W, axis=1)
        properties['row_sums_mean'] = np.mean(row_sums)
        properties['row_sums_std'] = np.std(row_sums)
        
        if np.any(np.abs(row_sums - 1.0) > 1e-10):
            max_deviation = np.max(np.abs(row_sums - 1.0))
            if max_deviation > 0.01:
                warnings.append(f"Row sums deviate from 1.0 (max deviation: {max_deviation:.6f})")
        
        # Check for isolated regions (zero row sums)
        isolated_regions = np.sum(row_sums < 1e-10)
        properties['isolated_regions'] = isolated_regions
        if isolated_regions > 0:
            warnings.append(f"{isolated_regions} regions have no spatial connections")
        
        # Compute eigenvalues
        try:
            eigenvals = np.linalg.eigvals(W)
            properties['max_eigenvalue'] = np.max(np.real(eigenvals))
            properties['min_eigenvalue'] = np.min(np.real(eigenvals))
            properties['spectral_radius'] = np.max(np.abs(eigenvals))
            
            # Check spectral radius for stability
            if properties['spectral_radius'] > 1.0:
                warnings.append(f"Spectral radius > 1.0 ({properties['spectral_radius']:.4f})")
        
        except np.linalg.LinAlgError:
            errors.append("Failed to compute eigenvalues")
        
        # Check connectivity
        try:
            # Simple connectivity check: can reach all regions in n-1 steps
            W_power = W.copy()
            connectivity_matrix = W.copy()
            
            for _ in range(self.n_regions - 1):
                W_power = W_power @ W
                connectivity_matrix += W_power
            
            # Check if all off-diagonal elements are positive
            np.fill_diagonal(connectivity_matrix, 0)
            connected_pairs = np.sum(connectivity_matrix > 1e-10)
            total_pairs = self.n_regions * (self.n_regions - 1)
            properties['connectivity_ratio'] = connected_pairs / total_pairs
            
            if connected_pairs < total_pairs:
                warnings.append(f"Matrix is not fully connected ({connected_pairs}/{total_pairs} pairs)")
        
        except Exception as e:
            warnings.append(f"Could not check connectivity: {str(e)}")
        
        # Overall validation status
        is_valid = len(errors) == 0
        
        return ValidationReport(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            properties=properties
        )
    
    def compute_spatial_lags(self, data: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
        """
        Compute spatial lags of regional data.
        
        Args:
            data: Regional data with regions as columns or index
            W: Spatial weight matrix
            
        Returns:
            DataFrame with spatial lags
        """
        if W.shape[0] != self.n_regions:
            raise SpatialModelError("Weight matrix size doesn't match number of regions")
        
        # Ensure data has correct regional dimension
        if data.shape[1] == self.n_regions:
            # Data has regions as columns
            spatial_lags = data.values @ W.T
            return pd.DataFrame(spatial_lags, index=data.index, columns=data.columns)
        
        elif data.shape[0] == self.n_regions:
            # Data has regions as rows
            spatial_lags = W @ data.values
            return pd.DataFrame(spatial_lags, index=data.index, columns=data.columns)
        
        else:
            raise SpatialModelError(
                f"Data shape {data.shape} doesn't match number of regions {self.n_regions}"
            )
    
    def test_spatial_autocorrelation(
        self, 
        residuals: pd.DataFrame, 
        W: np.ndarray,
        test_type: str = "moran"
    ) -> Dict[str, float]:
        """
        Test for spatial autocorrelation in residuals.
        
        Args:
            residuals: Residuals from econometric estimation
            W: Spatial weight matrix
            test_type: Type of test ('moran', 'lm_lag', 'lm_error')
            
        Returns:
            Dictionary with test statistics and p-values
        """
        if W.shape[0] != self.n_regions:
            raise SpatialModelError("Weight matrix size doesn't match number of regions")
        
        results = {}
        
        if test_type == "moran" or test_type == "all":
            # Moran's I test for each time period
            moran_stats = []
            
            for t in range(residuals.shape[0]):
                residual_vector = residuals.iloc[t].values
                
                # Compute Moran's I statistic
                n = len(residual_vector)
                W_sum = np.sum(W)
                
                if W_sum == 0:
                    moran_i = 0
                else:
                    # Center the residuals
                    residual_centered = residual_vector - np.mean(residual_vector)
                    
                    # Compute Moran's I
                    numerator = np.sum(W * np.outer(residual_centered, residual_centered))
                    denominator = np.sum(residual_centered ** 2)
                    
                    if denominator > 0:
                        moran_i = (n / W_sum) * (numerator / denominator)
                    else:
                        moran_i = 0
                
                moran_stats.append(moran_i)
            
            results['moran_i_mean'] = np.mean(moran_stats)
            results['moran_i_std'] = np.std(moran_stats)
            
            # Expected value and variance under null hypothesis
            expected_i = -1 / (n - 1)
            results['moran_i_expected'] = expected_i
            
            # Simple z-test (more sophisticated tests would require additional calculations)
            if len(moran_stats) > 1:
                z_stat = (results['moran_i_mean'] - expected_i) / (results['moran_i_std'] / np.sqrt(len(moran_stats)))
                results['moran_z_statistic'] = z_stat
        
        return results
    
    def create_distance_matrix(
        self, 
        coordinates: pd.DataFrame,
        distance_type: str = "euclidean"
    ) -> np.ndarray:
        """
        Create distance matrix from regional coordinates.
        
        Args:
            coordinates: DataFrame with 'latitude' and 'longitude' columns
            distance_type: Type of distance ('euclidean', 'manhattan', 'haversine')
            
        Returns:
            Distance matrix
        """
        if len(coordinates) != self.n_regions:
            raise SpatialModelError("Coordinates must be provided for all regions")
        
        required_cols = ['latitude', 'longitude']
        if not all(col in coordinates.columns for col in required_cols):
            raise SpatialModelError(f"Coordinates must contain columns: {required_cols}")
        
        coords = coordinates[required_cols].values
        
        if distance_type == "euclidean":
            distances = squareform(pdist(coords, metric='euclidean'))
        
        elif distance_type == "manhattan":
            distances = squareform(pdist(coords, metric='cityblock'))
        
        elif distance_type == "haversine":
            # Haversine distance for geographic coordinates
            distances = self._haversine_distance_matrix(coords)
        
        else:
            raise SpatialModelError(f"Unknown distance type: {distance_type}")
        
        return distances
    
    def _haversine_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute haversine distance matrix for geographic coordinates."""
        n = len(coords)
        distances = np.zeros((n, n))
        
        # Earth radius in kilometers
        R = 6371.0
        
        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = np.radians(coords[i])
                lat2, lon2 = np.radians(coords[j])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = (np.sin(dlat/2)**2 + 
                     np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
                c = 2 * np.arcsin(np.sqrt(a))
                
                distance = R * c
                distances[i, j] = distance
                distances[j, i] = distance
        
        return distances