"""
Integration tests for complete analysis workflows.

This module tests end-to-end workflows from data retrieval through
policy analysis and reporting, ensuring all components work together correctly.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

from regional_monetary_policy.data.fred_client import FREDClient
from regional_monetary_policy.data.data_manager import DataManager
from regional_monetary_policy.econometric.spatial_handler import SpatialModelHandler
from regional_monetary_policy.econometric.parameter_estimator import ParameterEstimator
from regional_monetary_policy.policy.mistake_decomposer import PolicyMistakeDecomposer
from regional_monetary_policy.policy.optimal_policy import OptimalPolicyCalculator
from regional_monetary_policy.policy.counterfactual_engine import CounterfactualEngine
from regional_monetary_policy.presentation.visualizers import RegionalMapVisualizer
from regional_monetary_policy.presentation.report_generator import ReportGenerator
from regional_monetary_policy.config.config_manager import ConfigManager
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters


class TestEndToEndWorkflow:
    """Test complete end-to-end analysis workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        temp_dir = tempfile.mkdtemp(prefix="rmp_integration_")
        workspace = Path(temp_dir)
        
        # Create directory structure
        (workspace / "data").mkdir()
        (workspace / "cache").mkdir()
        (workspace / "output").mkdir()
        (workspace / "config").mkdir()
        
        yield workspace
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_fred_client(self):
        """Mock FRED client for integration testing."""
        client = Mock(spec=FREDClient)
        
        # Mock successful API responses
        def mock_get_regional_series(series_codes, start_date, end_date, **kwargs):
            n_periods = 120  # 10 years of monthly data
            dates = pd.date_range(start_date, periods=n_periods, freq='ME')
            
            data = {}
            for i, series in enumerate(series_codes):
                # Generate realistic-looking economic data
                trend = 0.02 * np.arange(n_periods) / 12  # 2% annual trend
                cycle = 0.01 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Annual cycle
                noise = np.random.normal(0, 0.005, n_periods)
                
                if 'GDP' in series or 'OUTPUT' in series:
                    base_level = 100 + i * 10
                    values = base_level + trend + cycle + noise
                elif 'CPI' in series or 'INFLATION' in series:
                    base_level = 0.02 + i * 0.001  # 2% base inflation
                    values = base_level + 0.1 * trend + 0.5 * cycle + 0.1 * noise
                else:  # Interest rates
                    base_level = 0.03 + i * 0.002
                    values = base_level + 0.2 * trend + 0.3 * cycle + 0.05 * noise
                
                data[series] = pd.Series(values, index=dates)
            
            return pd.DataFrame(data)
        
        client.get_regional_series.side_effect = mock_get_regional_series
        client.validate_api_key.return_value = True
        
        return client
    
    @pytest.fixture
    def sample_config(self, temp_workspace):
        """Create sample configuration for testing."""
        config_data = {
            'data': {
                'regions': ['NY', 'CA', 'TX', 'FL'],
                'series_mapping': {
                    'output_gap': ['NYRGSP', 'CARGSP', 'TXRGSP', 'FLRGSP'],
                    'inflation': ['NYCPILFESL', 'CACPILFESL', 'TXCPILFESL', 'FLCPILFESL'],
                    'interest_rate': 'FEDFUNDS'
                },
                'start_date': '2010-01-01',
                'end_date': '2020-12-31'
            },
            'estimation': {
                'identification_strategy': 'baseline',
                'spatial_weight_method': 'trade_migration',
                'convergence_tolerance': 1e-6,
                'max_iterations': 1000,
                'bootstrap_replications': 50
            },
            'policy': {
                'welfare_weights': [0.3, 0.3, 0.25, 0.15],
                'loss_function': 'quadratic'
            },
            'output': {
                'save_intermediate': True,
                'generate_plots': True,
                'export_formats': ['csv', 'json']
            }
        }
        
        config_manager = ConfigManager(temp_workspace / "config")
        config_manager.save_config("integration_test", config_data)
        
        return config_manager.load_config("integration_test")
    
    @pytest.mark.integration
    def test_complete_analysis_workflow(self, temp_workspace, mock_fred_client, sample_config):
        """Test complete analysis workflow from data to results."""
        
        # Step 1: Data Management
        data_manager = DataManager(
            fred_client=mock_fred_client,
            cache_dir=temp_workspace / "cache"
        )
        
        # Load regional data
        regional_data = data_manager.load_regional_data(
            regions=sample_config['data']['regions'],
            indicators=['output_gap', 'inflation'],
            start_date=sample_config['data']['start_date'],
            end_date=sample_config['data']['end_date']
        )
        
        # Validate data structure
        assert isinstance(regional_data, RegionalDataset)
        assert regional_data.n_regions == 4
        assert regional_data.n_periods > 100
        
        # Step 2: Spatial Modeling
        spatial_handler = SpatialModelHandler(sample_config['data']['regions'])
        
        # Create mock spatial data
        trade_data = self._create_mock_trade_data(sample_config['data']['regions'])
        migration_data = self._create_mock_migration_data(sample_config['data']['regions'])
        financial_data = self._create_mock_financial_data(sample_config['data']['regions'])
        distance_matrix = self._create_distance_matrix(len(sample_config['data']['regions']))
        
        # Construct spatial weights
        spatial_weights = spatial_handler.construct_weights(
            trade_data, migration_data, financial_data, distance_matrix,
            weights=(0.4, 0.3, 0.2, 0.1)
        )
        
        # Validate spatial weights
        validation_report = spatial_handler.validate_spatial_matrix(spatial_weights)
        assert validation_report.is_valid
        
        # Step 3: Parameter Estimation
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        estimation_config = create_default_estimation_config()
        estimation_config.bootstrap_replications = 10  # Reduce for testing speed
        
        parameter_estimator = ParameterEstimator(spatial_handler, estimation_config)
        
        # Run full estimation
        estimation_results = parameter_estimator.estimate_full_model(regional_data)
        
        # Validate estimation results
        assert isinstance(estimation_results.regional_parameters, RegionalParameters)
        assert len(estimation_results.regional_parameters.sigma) == 4
        assert estimation_results.estimation_time > 0
        
        # Step 4: Policy Analysis
        welfare_weights = np.array(sample_config['policy']['welfare_weights'])
        
        # Optimal policy calculation
        optimal_calculator = OptimalPolicyCalculator(
            estimation_results.regional_parameters, 
            welfare_weights
        )
        
        # Create sample policy scenario
        sample_conditions = pd.DataFrame({
            'output_gap': [0.01, -0.005, 0.008, 0.002],
            'inflation': [0.02, 0.018, 0.022, 0.019],
            'expected_inflation': [0.02, 0.02, 0.02, 0.02]
        })
        
        optimal_rate = optimal_calculator.compute_optimal_rate(sample_conditions)
        assert isinstance(optimal_rate, (float, np.floating))
        
        # Policy mistake decomposition
        decomposer = PolicyMistakeDecomposer(
            estimation_results.regional_parameters,
            welfare_weights
        )
        
        # Mock Fed policy rate
        actual_rate = 0.025
        
        # Decompose policy mistake
        mistake_components = decomposer.decompose_policy_mistake(
            actual_rate=actual_rate,
            optimal_rate=optimal_rate,
            regional_conditions=sample_conditions,
            real_time_data=sample_conditions,  # Simplified for testing
            true_data=sample_conditions
        )
        
        # Validate decomposition
        assert hasattr(mistake_components, 'total_mistake')
        assert hasattr(mistake_components, 'information_effect')
        
        # Step 5: Counterfactual Analysis
        counterfactual_engine = CounterfactualEngine(
            estimation_results.regional_parameters,
            welfare_weights
        )
        
        # Generate scenarios
        baseline_scenario = counterfactual_engine.generate_baseline_scenario(regional_data)
        perfect_info_scenario = counterfactual_engine.generate_perfect_info_scenario(regional_data)
        
        # Validate scenarios
        assert baseline_scenario.name == 'baseline'
        assert perfect_info_scenario.name == 'perfect_info'
        assert len(baseline_scenario.policy_rates) > 0
        
        # Step 6: Visualization and Reporting
        visualizer = RegionalVisualizer()
        
        # Create sample plots
        regional_map = visualizer.create_regional_map(
            regional_data.output_gaps.iloc[:, -1],  # Latest period
            title="Regional Output Gaps"
        )
        assert regional_map is not None
        
        # Generate report
        report_generator = ReportGenerator(temp_workspace / "output")
        
        report_data = {
            'estimation_results': estimation_results,
            'policy_analysis': {
                'optimal_rate': optimal_rate,
                'mistake_decomposition': mistake_components
            },
            'counterfactual_results': {
                'baseline': baseline_scenario,
                'perfect_info': perfect_info_scenario
            }
        }
        
        report_path = report_generator.generate_comprehensive_report(
            report_data, 
            "integration_test_report"
        )
        
        # Validate report generation
        assert report_path.exists()
        assert report_path.suffix == '.pdf'
        
        # Step 7: Data Export
        export_dir = temp_workspace / "output" / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # Export estimation results
        estimation_results.regional_parameters.to_csv(export_dir / "parameters.csv")
        
        # Export policy analysis
        mistake_df = pd.DataFrame([{
            'total_mistake': mistake_components.total_mistake,
            'information_effect': mistake_components.information_effect,
            'optimal_rate': optimal_rate,
            'actual_rate': actual_rate
        }])
        mistake_df.to_csv(export_dir / "policy_analysis.csv", index=False)
        
        # Validate exports
        assert (export_dir / "parameters.csv").exists()
        assert (export_dir / "policy_analysis.csv").exists()
    
    def _create_mock_trade_data(self, regions):
        """Create mock trade flow data."""
        trade_flows = []
        for i, origin in enumerate(regions):
            for j, destination in enumerate(regions):
                if i != j:  # No self-trade
                    flow = np.random.uniform(50, 200)
                    trade_flows.append({
                        'origin': origin,
                        'destination': destination,
                        'trade_flow': flow
                    })
        return pd.DataFrame(trade_flows)
    
    def _create_mock_migration_data(self, regions):
        """Create mock migration flow data."""
        migration_flows = []
        for i, origin in enumerate(regions):
            for j, destination in enumerate(regions):
                if i != j:  # No self-migration
                    flow = np.random.uniform(10, 50)
                    migration_flows.append({
                        'origin': origin,
                        'destination': destination,
                        'migration_flow': flow
                    })
        return pd.DataFrame(migration_flows)
    
    def _create_mock_financial_data(self, regions):
        """Create mock financial flow data."""
        financial_flows = []
        for i, origin in enumerate(regions):
            for j, destination in enumerate(regions):
                if i != j and np.random.random() > 0.5:  # Sparse financial connections
                    flow = np.random.uniform(20, 100)
                    financial_flows.append({
                        'region1': origin,
                        'region2': destination,
                        'financial_linkage': flow
                    })
        return pd.DataFrame(financial_flows)
    
    def _create_distance_matrix(self, n_regions):
        """Create mock distance matrix."""
        # Create symmetric distance matrix
        distances = np.random.uniform(100, 1000, (n_regions, n_regions))
        distances = (distances + distances.T) / 2  # Make symmetric
        np.fill_diagonal(distances, 0)  # Zero diagonal
        return distances


class TestDataPipelineIntegration:
    """Test data pipeline integration from FRED to analysis-ready datasets."""
    
    @pytest.fixture
    def mock_fred_responses(self):
        """Mock FRED API responses for different data types."""
        responses = {}
        
        # GDP data
        responses['GDP'] = {
            'realtime_start': '2010-01-01',
            'realtime_end': '2020-12-31',
            'observations': [
                {'date': '2010-01-01', 'value': '100.0'},
                {'date': '2010-02-01', 'value': '100.5'},
                {'date': '2010-03-01', 'value': '101.0'},
            ]
        }
        
        # CPI data
        responses['CPI'] = {
            'realtime_start': '2010-01-01',
            'realtime_end': '2020-12-31',
            'observations': [
                {'date': '2010-01-01', 'value': '218.0'},
                {'date': '2010-02-01', 'value': '218.2'},
                {'date': '2010-03-01', 'value': '218.5'},
            ]
        }
        
        return responses
    
    @pytest.mark.integration
    def test_data_retrieval_and_processing(self, mock_fred_responses):
        """Test data retrieval and processing pipeline."""
        
        with patch('regional_monetary_policy.data.fred_client.requests.get') as mock_get:
            # Mock API responses
            mock_get.return_value.json.side_effect = lambda: mock_fred_responses['GDP']
            mock_get.return_value.status_code = 200
            
            # Create FRED client
            fred_client = FREDClient(api_key="test_key")
            
            # Test data retrieval
            series_data = fred_client.get_regional_series(
                ['NYRGSP'], '2010-01-01', '2010-03-01'
            )
            
            # Validate data structure
            assert isinstance(series_data, pd.DataFrame)
            assert len(series_data) == 3
            assert 'NYRGSP' in series_data.columns
    
    @pytest.mark.integration
    def test_data_validation_pipeline(self):
        """Test data validation and quality checking pipeline."""
        
        # Create sample data with quality issues
        dates = pd.date_range('2010-01-01', periods=100, freq='ME')
        
        # Data with missing values and outliers
        output_data = np.random.normal(100, 5, 100)
        output_data[10:15] = np.nan  # Missing values
        output_data[50] = 200  # Outlier
        
        inflation_data = np.random.normal(0.02, 0.005, 100)
        inflation_data[20:25] = np.nan  # Missing values
        inflation_data[75] = 0.1  # Outlier
        
        # Create dataset
        regional_data = RegionalDataset(
            output_gaps=pd.DataFrame({
                'Region_1': output_data,
                'Region_2': np.random.normal(100, 5, 100)
            }, index=dates),
            inflation_rates=pd.DataFrame({
                'Region_1': inflation_data,
                'Region_2': np.random.normal(0.02, 0.005, 100)
            }, index=dates),
            interest_rates=pd.Series(np.random.normal(0.03, 0.01, 100), index=dates),
            real_time_estimates={},
            metadata={}
        )
        
        # Test data validation
        from regional_monetary_policy.data_quality import DataQualityChecker
        quality_checker = DataQualityChecker()
        
        quality_report = quality_checker.check_data_quality(regional_data)
        
        # Should detect missing values and outliers
        assert quality_report.has_missing_values
        assert quality_report.has_outliers
        assert len(quality_report.recommendations) > 0


class TestEstimationPipelineIntegration:
    """Test integration of estimation pipeline components."""
    
    @pytest.mark.integration
    def test_three_stage_estimation_integration(self):
        """Test integration of three-stage estimation procedure."""
        
        # Generate synthetic data for testing
        n_regions = 3
        n_periods = 100
        
        # Create synthetic regional dataset
        synthetic_data = self._generate_synthetic_dataset(n_regions, n_periods)
        
        # Create spatial handler
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        spatial_handler = SpatialModelHandler(regions)
        
        # Create estimation configuration
        from regional_monetary_policy.econometric.parameter_estimator import create_default_estimation_config
        config = create_default_estimation_config()
        config.bootstrap_replications = 5  # Reduce for testing
        
        # Create parameter estimator
        estimator = ParameterEstimator(spatial_handler, config)
        
        # Test Stage 1: Spatial weights
        spatial_results = estimator.estimate_stage_one(synthetic_data)
        assert spatial_results.weight_matrix.shape == (n_regions, n_regions)
        
        # Test Stage 2: Regional parameters
        regional_params = estimator.estimate_stage_two(
            synthetic_data, spatial_results.weight_matrix
        )
        assert len(regional_params.sigma) == n_regions
        
        # Test Stage 3: Policy parameters
        policy_params = estimator.estimate_stage_three(synthetic_data, regional_params)
        assert 'inflation_coefficient' in policy_params
        
        # Test full estimation
        full_results = estimator.estimate_full_model(synthetic_data)
        assert full_results.estimation_time > 0
        assert len(estimator.stage_results) == 3
    
    def _generate_synthetic_dataset(self, n_regions, n_periods):
        """Generate synthetic dataset for testing."""
        dates = pd.date_range('2010-01-01', periods=n_periods, freq='ME')
        regions = [f"Region_{i+1}" for i in range(n_regions)]
        
        # Generate correlated regional data
        np.random.seed(42)
        
        # Output gaps
        output_gaps = np.random.multivariate_normal(
            mean=np.zeros(n_regions),
            cov=0.01 * (np.eye(n_regions) + 0.3 * np.ones((n_regions, n_regions))),
            size=n_periods
        ).T
        
        # Inflation rates
        inflation_rates = np.random.multivariate_normal(
            mean=0.02 * np.ones(n_regions),
            cov=0.0001 * (np.eye(n_regions) + 0.2 * np.ones((n_regions, n_regions))),
            size=n_periods
        ).T
        
        # Interest rates (common monetary policy)
        interest_rates = np.random.normal(0.03, 0.005, n_periods)
        
        return RegionalDataset(
            output_gaps=pd.DataFrame(output_gaps, index=regions, columns=dates),
            inflation_rates=pd.DataFrame(inflation_rates, index=regions, columns=dates),
            interest_rates=pd.Series(interest_rates, index=dates),
            real_time_estimates={},
            metadata={'synthetic': True}
        )


class TestPolicyAnalysisPipelineIntegration:
    """Test integration of policy analysis pipeline."""
    
    @pytest.mark.integration
    def test_policy_analysis_workflow(self):
        """Test complete policy analysis workflow integration."""
        
        # Create sample regional parameters
        regional_params = RegionalParameters(
            sigma=np.array([0.5, 0.7, 0.6]),
            kappa=np.array([0.3, 0.4, 0.35]),
            psi=np.array([0.1, 0.15, 0.12]),
            phi=np.array([0.05, 0.08, 0.06]),
            beta=np.array([0.99, 0.99, 0.99]),
            standard_errors={
                'sigma': np.array([0.05, 0.07, 0.06]),
                'kappa': np.array([0.03, 0.04, 0.035])
            },
            confidence_intervals={}
        )
        
        # Welfare weights
        welfare_weights = np.array([0.4, 0.35, 0.25])
        
        # Test optimal policy calculation
        optimal_calculator = OptimalPolicyCalculator(regional_params, welfare_weights)
        
        regional_conditions = pd.DataFrame({
            'output_gap': [0.01, -0.005, 0.008],
            'inflation': [0.02, 0.018, 0.022],
            'expected_inflation': [0.02, 0.02, 0.02]
        })
        
        optimal_rate = optimal_calculator.compute_optimal_rate(regional_conditions)
        assert isinstance(optimal_rate, (float, np.floating))
        
        # Test policy mistake decomposition
        decomposer = PolicyMistakeDecomposer(regional_params, welfare_weights)
        
        mistake_components = decomposer.decompose_policy_mistake(
            actual_rate=0.025,
            optimal_rate=optimal_rate,
            regional_conditions=regional_conditions,
            real_time_data=regional_conditions,
            true_data=regional_conditions
        )
        
        # Validate decomposition components
        assert hasattr(mistake_components, 'total_mistake')
        assert hasattr(mistake_components, 'information_effect')
        assert hasattr(mistake_components, 'weight_misallocation_effect')
        
        # Test counterfactual analysis
        counterfactual_engine = CounterfactualEngine(regional_params, welfare_weights)
        
        # Create sample historical data
        n_periods = 50
        dates = pd.date_range('2015-01-01', periods=n_periods, freq='ME')
        
        historical_data = RegionalDataset(
            output_gaps=pd.DataFrame(
                np.random.normal(0, 0.01, (3, n_periods)),
                index=['Region_1', 'Region_2', 'Region_3'],
                columns=dates
            ),
            inflation_rates=pd.DataFrame(
                np.random.normal(0.02, 0.005, (3, n_periods)),
                index=['Region_1', 'Region_2', 'Region_3'],
                columns=dates
            ),
            interest_rates=pd.Series(
                np.random.normal(0.03, 0.01, n_periods),
                index=dates
            ),
            real_time_estimates={},
            metadata={}
        )
        
        # Generate scenarios
        baseline_scenario = counterfactual_engine.generate_baseline_scenario(historical_data)
        perfect_info_scenario = counterfactual_engine.generate_perfect_info_scenario(historical_data)
        
        # Validate scenarios
        assert baseline_scenario.name == 'baseline'
        assert perfect_info_scenario.name == 'perfect_info'
        assert len(baseline_scenario.policy_rates) == n_periods
        assert len(perfect_info_scenario.policy_rates) == n_periods


if __name__ == '__main__':
    pytest.main([__file__])