"""
Unit tests for spatial modeling infrastructure.

Tests the SpatialModelHandler class and related functionality for constructing,
validating, and working with spatial weight matrices.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from regional_monetary_policy.econometric.spatial_handler import (
    SpatialModelHandler,
    SpatialWeightResults,
    ValidationReport
)
from regional_monetary_policy.exceptions import SpatialModelError


class TestSpatialModelHandler:
    """Test cases for SpatialModelHandler class."""
    
    @pytest.fixture
    def regions(self):
        """Sample region list for testing."""
        return ['NY', 'CA', 'TX', 'FL']
    
    @pytest.fixture
    def handler(self, regions):
        """SpatialModelHandler instance for testing."""
        return SpatialModelHandler(regions)
    
    @pytest.fixture
    def sample_trade_data(self):
        """Sample trade flow data."""
        return pd.DataFrame({
            'origin': ['NY', 'NY', 'CA', 'CA', 'TX', 'TX', 'FL', 'FL'],
            'destination': ['CA', 'TX', 'NY', 'FL', 'NY', 'FL', 'CA', 'TX'],
            'trade_flow': [100, 80, 90, 70, 60, 50, 40, 30]
        })
    
    @pytest.fixture
    def sample_migration_data(self):
        """Sample migration flow data."""
        return pd.DataFrame({
            'origin': ['NY', 'NY', 'CA', 'CA', 'TX', 'TX', 'FL', 'FL'],
            'destination': ['CA', 'TX', 'NY', 'FL', 'NY', 'FL', 'CA', 'TX'],
            'migration_flow': [50, 40, 45, 35, 30, 25, 20, 15]
        })
    
    @pytest.fixture
    def sample_financial_data(self):
        """Sample financial linkage data."""
        return pd.DataFrame({
            'region1': ['NY', 'NY', 'NY', 'CA', 'CA', 'TX'],
            'region2': ['CA', 'TX', 'FL', 'TX', 'FL', 'FL'],
            'financial_linkage': [0.8, 0.6, 0.7, 0.5, 0.4, 0.3]
        })
    
    @pytest.fixture
    def sample_distance_matrix(self):
        """Sample distance matrix."""
        return np.array([
            [0, 2500, 1400, 1100],  # NY distances
            [2500, 0, 1200, 2400],  # CA distances
            [1400, 1200, 0, 900],   # TX distances
            [1100, 2400, 900, 0]    # FL distances
        ])
    
    @pytest.fixture
    def sample_coordinates(self):
        """Sample regional coordinates."""
        return pd.DataFrame({
            'latitude': [40.7128, 34.0522, 29.7604, 25.7617],
            'longitude': [-74.0060, -118.2437, -95.3698, -80.1918]
        }, index=['NY', 'CA', 'TX', 'FL'])
    
    def test_initialization(self, regions):
        """Test SpatialModelHandler initialization."""
        handler = SpatialModelHandler(regions)
        
        assert handler.regions == regions
        assert handler.n_regions == len(regions)
        assert handler.region_index == {'NY': 0, 'CA': 1, 'TX': 2, 'FL': 3}
    
    def test_initialization_insufficient_regions(self):
        """Test initialization with insufficient regions."""
        with pytest.raises(SpatialModelError, match="At least 2 regions required"):
            SpatialModelHandler(['NY'])
    
    def test_construct_weights_basic(self, handler, sample_trade_data, sample_distance_matrix):
        """Test basic spatial weight matrix construction."""
        result = handler.construct_weights(
            trade_data=sample_trade_data,
            distance_matrix=sample_distance_matrix,
            weights=(0.7, 0.0, 0.0, 0.3)
        )
        
        assert isinstance(result, SpatialWeightResults)
        assert result.weight_matrix.shape == (4, 4)
        assert 'trade' in result.component_matrices
        assert 'distance' in result.component_matrices
        assert result.component_weights['trade'] == 0.7
        assert result.component_weights['distance'] == 0.3
    
    def test_construct_weights_all_components(
        self, 
        handler, 
        sample_trade_data, 
        sample_migration_data,
        sample_financial_data,
        sample_distance_matrix
    ):
        """Test spatial weight construction with all components."""
        result = handler.construct_weights(
            trade_data=sample_trade_data,
            migration_data=sample_migration_data,
            financial_data=sample_financial_data,
            distance_matrix=sample_distance_matrix,
            weights=(0.4, 0.3, 0.2, 0.1)
        )
        
        assert result.weight_matrix.shape == (4, 4)
        assert len(result.component_matrices) == 4
        assert all(comp in result.component_matrices for comp in ['trade', 'migration', 'financial', 'distance'])
    
    def test_construct_weights_invalid_weights(self, handler):
        """Test construction with invalid weight parameters."""
        with pytest.raises(SpatialModelError, match="Must provide exactly 4 weights"):
            handler.construct_weights(weights=(0.5, 0.5))
        
        with pytest.raises(SpatialModelError, match="Component weights must sum to 1.0"):
            handler.construct_weights(weights=(0.3, 0.3, 0.3, 0.3))
    
    def test_process_trade_data(self, handler, sample_trade_data):
        """Test trade data processing."""
        trade_matrix = handler._process_trade_data(sample_trade_data)
        
        assert trade_matrix.shape == (4, 4)
        assert np.all(trade_matrix >= 0)
        
        # Check row normalization (approximately)
        row_sums = np.sum(trade_matrix, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)
    
    def test_process_trade_data_missing_columns(self, handler):
        """Test trade data processing with missing columns."""
        invalid_data = pd.DataFrame({
            'origin': ['NY', 'CA'],
            'destination': ['CA', 'NY']
            # Missing 'trade_flow' column
        })
        
        with pytest.raises(SpatialModelError, match="Trade data must contain columns"):
            handler._process_trade_data(invalid_data)
    
    def test_process_migration_data(self, handler, sample_migration_data):
        """Test migration data processing."""
        migration_matrix = handler._process_migration_data(sample_migration_data)
        
        assert migration_matrix.shape == (4, 4)
        assert np.all(migration_matrix >= 0)
        
        # Check row normalization
        row_sums = np.sum(migration_matrix, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)
    
    def test_process_financial_data(self, handler, sample_financial_data):
        """Test financial data processing."""
        financial_matrix = handler._process_financial_data(sample_financial_data)
        
        assert financial_matrix.shape == (4, 4)
        assert np.all(financial_matrix >= 0)
        
        # Check that the matrix is approximately symmetric (within reasonable tolerance)
        # After normalization, perfect symmetry may not be preserved
        assert np.allclose(financial_matrix, financial_matrix.T, atol=1e-6)
    
    def test_process_distance_matrix(self, handler, sample_distance_matrix):
        """Test distance matrix processing."""
        distance_weights = handler._process_distance_matrix(sample_distance_matrix)
        
        assert distance_weights.shape == (4, 4)
        assert np.all(distance_weights >= 0)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(distance_weights), 0, atol=1e-10)
        
        # Check row normalization
        row_sums = np.sum(distance_weights, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)
    
    def test_process_distance_matrix_wrong_shape(self, handler):
        """Test distance matrix processing with wrong shape."""
        wrong_shape_matrix = np.random.rand(3, 3)
        
        with pytest.raises(SpatialModelError, match="Distance matrix shape"):
            handler._process_distance_matrix(wrong_shape_matrix)
    
    def test_normalize_matrix_row(self, handler):
        """Test row normalization."""
        matrix = np.array([
            [0, 2, 1, 1],
            [1, 0, 2, 1],
            [1, 1, 0, 2],
            [2, 1, 1, 0]
        ])
        
        normalized = handler._normalize_matrix(matrix, method="row")
        
        # Check row sums equal 1
        row_sums = np.sum(normalized, axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_normalize_matrix_spectral(self, handler):
        """Test spectral normalization."""
        matrix = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        normalized = handler._normalize_matrix(matrix, method="spectral")
        
        # Check largest eigenvalue is approximately 1
        eigenvals = np.linalg.eigvals(normalized)
        max_eigenval = np.max(np.real(eigenvals))
        assert max_eigenval <= 1.0 + 1e-10
    
    def test_normalize_matrix_none(self, handler):
        """Test no normalization."""
        matrix = np.random.rand(4, 4)
        normalized = handler._normalize_matrix(matrix, method="none")
        
        assert np.allclose(matrix, normalized)
    
    def test_normalize_matrix_invalid_method(self, handler):
        """Test normalization with invalid method."""
        matrix = np.random.rand(4, 4)
        
        with pytest.raises(SpatialModelError, match="Unknown normalization method"):
            handler._normalize_matrix(matrix, method="invalid")
    
    def test_validate_spatial_matrix_valid(self, handler):
        """Test validation of a valid spatial weight matrix."""
        # Create a valid row-standardized matrix
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        report = handler.validate_spatial_matrix(W)
        
        assert report.is_valid
        assert len(report.errors) == 0
        assert 'diagonal_sum' in report.properties
        assert 'row_sums_mean' in report.properties
    
    def test_validate_spatial_matrix_wrong_shape(self, handler):
        """Test validation with wrong matrix shape."""
        W = np.random.rand(3, 3)  # Wrong size
        
        report = handler.validate_spatial_matrix(W)
        
        assert not report.is_valid
        assert any("Matrix shape" in error for error in report.errors)
    
    def test_validate_spatial_matrix_negative_entries(self, handler):
        """Test validation with negative entries."""
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, -0.1, 0.7],  # Negative entry
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        report = handler.validate_spatial_matrix(W)
        
        assert any("negative entries" in warning for warning in report.warnings)
    
    def test_validate_spatial_matrix_non_zero_diagonal(self, handler):
        """Test validation with non-zero diagonal."""
        W = np.array([
            [0.1, 0.4, 0.3, 0.2],  # Non-zero diagonal
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        report = handler.validate_spatial_matrix(W)
        
        assert any("diagonal elements" in warning for warning in report.warnings)
    
    def test_compute_spatial_lags_columns(self, handler):
        """Test spatial lag computation with regions as columns."""
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        # Data with regions as columns
        data = pd.DataFrame(
            np.random.rand(10, 4),
            columns=['NY', 'CA', 'TX', 'FL']
        )
        
        spatial_lags = handler.compute_spatial_lags(data, W)
        
        assert spatial_lags.shape == data.shape
        assert list(spatial_lags.columns) == list(data.columns)
    
    def test_compute_spatial_lags_rows(self, handler):
        """Test spatial lag computation with regions as rows."""
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        # Data with regions as rows
        data = pd.DataFrame(
            np.random.rand(4, 10),
            index=['NY', 'CA', 'TX', 'FL']
        )
        
        spatial_lags = handler.compute_spatial_lags(data, W)
        
        assert spatial_lags.shape == data.shape
        assert list(spatial_lags.index) == list(data.index)
    
    def test_compute_spatial_lags_wrong_dimensions(self, handler):
        """Test spatial lag computation with wrong data dimensions."""
        W = np.random.rand(4, 4)
        data = pd.DataFrame(np.random.rand(5, 5))  # Wrong dimensions
        
        with pytest.raises(SpatialModelError, match="Data shape"):
            handler.compute_spatial_lags(data, W)
    
    def test_test_spatial_autocorrelation_moran(self, handler):
        """Test Moran's I spatial autocorrelation test."""
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        # Create residuals with some spatial pattern
        residuals = pd.DataFrame(
            np.random.rand(10, 4),
            columns=['NY', 'CA', 'TX', 'FL']
        )
        
        results = handler.test_spatial_autocorrelation(residuals, W, test_type="moran")
        
        assert 'moran_i_mean' in results
        assert 'moran_i_std' in results
        assert 'moran_i_expected' in results
    
    def test_create_distance_matrix_euclidean(self, handler, sample_coordinates):
        """Test distance matrix creation with Euclidean distance."""
        distances = handler.create_distance_matrix(sample_coordinates, distance_type="euclidean")
        
        assert distances.shape == (4, 4)
        assert np.allclose(np.diag(distances), 0)  # Diagonal should be zero
        assert np.allclose(distances, distances.T)  # Should be symmetric
    
    def test_create_distance_matrix_manhattan(self, handler, sample_coordinates):
        """Test distance matrix creation with Manhattan distance."""
        distances = handler.create_distance_matrix(sample_coordinates, distance_type="manhattan")
        
        assert distances.shape == (4, 4)
        assert np.allclose(np.diag(distances), 0)
        assert np.allclose(distances, distances.T)
    
    def test_create_distance_matrix_haversine(self, handler, sample_coordinates):
        """Test distance matrix creation with Haversine distance."""
        distances = handler.create_distance_matrix(sample_coordinates, distance_type="haversine")
        
        assert distances.shape == (4, 4)
        assert np.allclose(np.diag(distances), 0)
        assert np.allclose(distances, distances.T)
        
        # Haversine distances should be reasonable for US cities (hundreds to thousands of km)
        assert np.all(distances >= 0)
        assert np.all(distances[distances > 0] > 100)  # At least 100 km between different cities
    
    def test_create_distance_matrix_wrong_regions(self, handler):
        """Test distance matrix creation with wrong number of regions."""
        coords = pd.DataFrame({
            'latitude': [40.7128, 34.0522],  # Only 2 regions
            'longitude': [-74.0060, -118.2437]
        })
        
        with pytest.raises(SpatialModelError, match="Coordinates must be provided for all regions"):
            handler.create_distance_matrix(coords)
    
    def test_create_distance_matrix_missing_columns(self, handler):
        """Test distance matrix creation with missing coordinate columns."""
        coords = pd.DataFrame({
            'latitude': [40.7128, 34.0522, 29.7604, 25.7617]
            # Missing longitude column
        })
        
        with pytest.raises(SpatialModelError, match="Coordinates must contain columns"):
            handler.create_distance_matrix(coords)
    
    def test_create_distance_matrix_invalid_type(self, handler, sample_coordinates):
        """Test distance matrix creation with invalid distance type."""
        with pytest.raises(SpatialModelError, match="Unknown distance type"):
            handler.create_distance_matrix(sample_coordinates, distance_type="invalid")


class TestSpatialWeightResults:
    """Test cases for SpatialWeightResults class."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample SpatialWeightResults for testing."""
        W = np.array([
            [0, 0.5, 0.3, 0.2],
            [0.4, 0, 0.4, 0.2],
            [0.3, 0.3, 0, 0.4],
            [0.2, 0.2, 0.6, 0]
        ])
        
        component_matrices = {
            'trade': np.random.rand(4, 4),
            'migration': np.random.rand(4, 4),
            'financial': np.random.rand(4, 4),
            'distance': np.random.rand(4, 4)
        }
        
        component_weights = {
            'trade': 0.4,
            'migration': 0.3,
            'financial': 0.2,
            'distance': 0.1
        }
        
        validation_report = ValidationReport(
            is_valid=True,
            warnings=[],
            errors=[],
            properties={}
        )
        
        return SpatialWeightResults(
            weight_matrix=W,
            component_matrices=component_matrices,
            component_weights=component_weights,
            validation_report=validation_report,
            construction_method="combined_row_normalized"
        )
    
    def test_get_eigenvalues(self, sample_results):
        """Test eigenvalue computation."""
        eigenvals = sample_results.get_eigenvalues()
        
        assert len(eigenvals) == 4
        assert np.all(np.isfinite(eigenvals))
    
    def test_get_spectral_radius(self, sample_results):
        """Test spectral radius computation."""
        spectral_radius = sample_results.get_spectral_radius()
        
        assert isinstance(spectral_radius, float)
        assert spectral_radius >= 0


class TestValidationReport:
    """Test cases for ValidationReport class."""
    
    def test_summary_valid_matrix(self):
        """Test summary generation for valid matrix."""
        report = ValidationReport(
            is_valid=True,
            warnings=["Minor warning"],
            errors=[],
            properties={'row_sums_mean': 1.0, 'spectral_radius': 0.8}
        )
        
        summary = report.summary()
        
        assert "VALID" in summary
        assert "Minor warning" in summary
        assert "row_sums_mean: 1.0" in summary
    
    def test_summary_invalid_matrix(self):
        """Test summary generation for invalid matrix."""
        report = ValidationReport(
            is_valid=False,
            warnings=[],
            errors=["Critical error"],
            properties={}
        )
        
        summary = report.summary()
        
        assert "INVALID" in summary
        assert "Critical error" in summary


class TestIntegration:
    """Integration tests for spatial modeling workflow."""
    
    def test_complete_workflow(self):
        """Test complete spatial modeling workflow."""
        # Setup
        regions = ['NY', 'CA', 'TX', 'FL']
        handler = SpatialModelHandler(regions)
        
        # Create sample data
        trade_data = pd.DataFrame({
            'origin': ['NY', 'CA', 'TX', 'FL'],
            'destination': ['CA', 'TX', 'FL', 'NY'],
            'trade_flow': [100, 90, 80, 70]
        })
        
        distance_matrix = np.array([
            [0, 2500, 1400, 1100],
            [2500, 0, 1200, 2400],
            [1400, 1200, 0, 900],
            [1100, 2400, 900, 0]
        ])
        
        # Construct spatial weights
        results = handler.construct_weights(
            trade_data=trade_data,
            distance_matrix=distance_matrix,
            weights=(0.7, 0.0, 0.0, 0.3)
        )
        
        # Validate results
        assert results.validation_report.is_valid
        assert results.weight_matrix.shape == (4, 4)
        
        # Test spatial lags
        data = pd.DataFrame(
            np.random.rand(10, 4),
            columns=regions
        )
        
        spatial_lags = handler.compute_spatial_lags(data, results.weight_matrix)
        assert spatial_lags.shape == data.shape
        
        # Test autocorrelation
        residuals = pd.DataFrame(
            np.random.rand(5, 4),
            columns=regions
        )
        
        autocorr_results = handler.test_spatial_autocorrelation(
            residuals, 
            results.weight_matrix
        )
        
        assert 'moran_i_mean' in autocorr_results
    
    def test_error_handling_workflow(self):
        """Test error handling in spatial modeling workflow."""
        regions = ['NY', 'CA']
        handler = SpatialModelHandler(regions)
        
        # Test with invalid trade data
        invalid_trade_data = pd.DataFrame({
            'origin': ['NY'],
            'destination': ['CA']
            # Missing trade_flow column
        })
        
        with pytest.raises(SpatialModelError):
            handler.construct_weights(trade_data=invalid_trade_data)
        
        # Test with wrong dimension distance matrix
        wrong_distance_matrix = np.random.rand(3, 3)
        
        with pytest.raises(SpatialModelError):
            handler.construct_weights(distance_matrix=wrong_distance_matrix)