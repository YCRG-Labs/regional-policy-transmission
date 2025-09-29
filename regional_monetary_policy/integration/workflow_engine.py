"""
Workflow Engine for orchestrating complete analysis pipelines.

This module provides the main interface for running end-to-end monetary policy
analysis workflows, from data retrieval to report generation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np

from ..data.fred_client import FREDClient
from ..data.data_manager import DataManager
from ..econometric.spatial_handler import SpatialModelHandler
from ..econometric.parameter_estimator import ParameterEstimator
from ..policy.mistake_decomposer import PolicyMistakeDecomposer
from ..policy.optimal_policy import OptimalPolicyCalculator
from ..policy.counterfactual_engine import CounterfactualEngine
from ..presentation.visualizers import (
    RegionalMapVisualizer, TimeSeriesVisualizer, ParameterVisualizer,
    PolicyAnalysisVisualizer, CounterfactualVisualizer
)
from ..presentation.report_generator import ReportGenerator
from ..config.config_manager import ConfigManager
from ..exceptions import RegionalMonetaryPolicyError
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """
    Main workflow engine that orchestrates complete analysis pipelines.
    
    This class provides high-level methods for running standard analysis workflows
    while handling component coordination, error recovery, and progress monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the workflow engine.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        setup_logging()
        logger.info("Initializing WorkflowEngine")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config(config_path)
        
        # Initialize core components
        self._initialize_components()
        
        # Track workflow state
        self.current_workflow = None
        self.workflow_results = {}
        
    def _initialize_components(self):
        """Initialize all system components with proper configuration."""
        try:
            # Data layer
            self.fred_client = FREDClient(
                api_key=self.config.data.fred_api_key,
                cache_dir=self.config.data.cache_directory
            )
            self.data_manager = DataManager(
                fred_client=self.fred_client
            )
            
            # Econometric layer
            self.spatial_handler = SpatialModelHandler(
                regions=self.config.data.regions
            )
            self.parameter_estimator = ParameterEstimator()
            
            # Policy analysis layer
            self.mistake_decomposer = PolicyMistakeDecomposer()
            self.optimal_policy_calc = OptimalPolicyCalculator()
            self.counterfactual_engine = CounterfactualEngine()
            
            # Presentation layer
            self.map_visualizer = RegionalMapVisualizer(self.config.data.regions)
            self.timeseries_visualizer = TimeSeriesVisualizer()
            self.parameter_visualizer = ParameterVisualizer()
            self.policy_visualizer = PolicyAnalysisVisualizer()
            self.counterfactual_visualizer = CounterfactualVisualizer()
            self.report_generator = ReportGenerator()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RegionalMonetaryPolicyError(f"Component initialization failed: {e}")
    
    def run_complete_analysis(self, 
                            start_date: str,
                            end_date: str,
                            regions: Optional[List[str]] = None,
                            analysis_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete end-to-end analysis workflow.
        
        Args:
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            regions: List of region codes. If None, uses config default.
            analysis_name: Name for this analysis run
            
        Returns:
            Dictionary containing all analysis results
        """
        if analysis_name is None:
            analysis_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.current_workflow = analysis_name
        logger.info(f"Starting complete analysis workflow: {analysis_name}")
        
        try:
            # Step 1: Data retrieval and preparation
            logger.info("Step 1: Data retrieval and preparation")
            regional_data = self._retrieve_and_prepare_data(start_date, end_date, regions)
            
            # Step 2: Spatial modeling and parameter estimation
            logger.info("Step 2: Spatial modeling and parameter estimation")
            spatial_weights, regional_params = self._estimate_parameters(regional_data)
            
            # Step 3: Policy analysis and mistake decomposition
            logger.info("Step 3: Policy analysis and mistake decomposition")
            policy_results = self._analyze_policy_mistakes(regional_data, regional_params)
            
            # Step 4: Counterfactual analysis
            logger.info("Step 4: Counterfactual analysis")
            counterfactual_results = self._run_counterfactual_analysis(
                regional_data, regional_params
            )
            
            # Step 5: Visualization and reporting
            logger.info("Step 5: Visualization and reporting")
            visualizations = self._generate_visualizations(
                regional_data, regional_params, policy_results, counterfactual_results
            )
            
            # Step 6: Report generation
            logger.info("Step 6: Report generation")
            report_path = self._generate_comprehensive_report(
                analysis_name, regional_data, regional_params, 
                policy_results, counterfactual_results, visualizations
            )
            
            # Compile final results
            results = {
                'analysis_name': analysis_name,
                'timestamp': datetime.now().isoformat(),
                'data': regional_data,
                'spatial_weights': spatial_weights,
                'regional_parameters': regional_params,
                'policy_analysis': policy_results,
                'counterfactual_analysis': counterfactual_results,
                'visualizations': visualizations,
                'report_path': report_path,
                'config': self.config
            }
            
            self.workflow_results[analysis_name] = results
            logger.info(f"Complete analysis workflow finished successfully: {analysis_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise RegionalMonetaryPolicyError(f"Analysis workflow failed: {e}")
    
    def _retrieve_and_prepare_data(self, start_date: str, end_date: str, 
                                 regions: Optional[List[str]]) -> Any:
        """Retrieve and prepare regional economic data."""
        if regions is None:
            regions = self.config.regions
            
        # Load regional data
        regional_data = self.data_manager.load_regional_data(
            regions=regions,
            indicators=['output_gap', 'inflation', 'interest_rate'],
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate data quality
        validation_report = self.data_manager.validate_data_quality(regional_data.to_dataframe())
        if not validation_report.is_valid:
            logger.warning(f"Data quality issues detected: {validation_report.issues}")
            
        # Handle missing data if needed
        if validation_report.has_missing_data:
            regional_data = self.data_manager.handle_missing_data(
                regional_data, method=self.config.missing_data_method
            )
            
        return regional_data
    
    def _estimate_parameters(self, regional_data: Any) -> Tuple[np.ndarray, Any]:
        """Estimate spatial weights and regional parameters."""
        # Construct spatial weight matrix
        spatial_weights = self.spatial_handler.construct_weights(
            trade_data=self.data_manager.get_trade_data(),
            migration_data=self.data_manager.get_migration_data(),
            financial_data=self.data_manager.get_financial_data(),
            distance_matrix=self.data_manager.get_distance_matrix(),
            weights=self.config.spatial_weights
        )
        
        # Validate spatial matrix
        validation_report = self.spatial_handler.validate_spatial_matrix(spatial_weights)
        if not validation_report.is_valid:
            raise RegionalMonetaryPolicyError(f"Invalid spatial matrix: {validation_report.issues}")
        
        # Estimate regional parameters
        self.parameter_estimator.set_spatial_weights(spatial_weights)
        regional_params = self.parameter_estimator.estimate_parameters(regional_data)
        
        # Run identification tests
        identification_report = self.parameter_estimator.run_identification_tests(regional_data)
        if not identification_report.is_identified:
            logger.warning(f"Identification issues: {identification_report.warnings}")
            
        return spatial_weights, regional_params
    
    def _analyze_policy_mistakes(self, regional_data: Any, regional_params: Any) -> Dict[str, Any]:
        """Analyze policy mistakes and decompose sources."""
        # Calculate optimal policy
        optimal_policy = self.optimal_policy_calc.compute_optimal_policy_path(
            regional_data, regional_params
        )
        
        # Get actual Fed policy
        actual_policy = regional_data.interest_rates
        
        # Decompose policy mistakes
        mistake_decomposition = self.mistake_decomposer.decompose_policy_mistakes(
            actual_policy=actual_policy,
            optimal_policy=optimal_policy,
            regional_data=regional_data,
            regional_params=regional_params
        )
        
        return {
            'optimal_policy': optimal_policy,
            'actual_policy': actual_policy,
            'mistake_decomposition': mistake_decomposition
        }
    
    def _run_counterfactual_analysis(self, regional_data: Any, 
                                   regional_params: Any) -> Dict[str, Any]:
        """Run counterfactual policy analysis."""
        # Generate all four policy scenarios
        scenarios = {}
        
        scenarios['baseline'] = self.counterfactual_engine.generate_baseline_scenario(
            regional_data, regional_params
        )
        
        scenarios['perfect_info'] = self.counterfactual_engine.generate_perfect_info_scenario(
            regional_data, regional_params
        )
        
        scenarios['optimal_regional'] = self.counterfactual_engine.generate_optimal_regional_scenario(
            regional_data, regional_params
        )
        
        scenarios['perfect_regional'] = self.counterfactual_engine.generate_perfect_regional_scenario(
            regional_data, regional_params
        )
        
        # Compare scenarios and compute welfare rankings
        comparison_results = self.counterfactual_engine.compare_scenarios(
            list(scenarios.values())
        )
        
        return {
            'scenarios': scenarios,
            'comparison_results': comparison_results
        }
    
    def _generate_visualizations(self, regional_data: Any, regional_params: Any,
                               policy_results: Dict[str, Any], 
                               counterfactual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all visualizations for the analysis."""
        visualizations = {}
        
        # Regional data visualizations
        visualizations['regional_maps'] = self.map_visualizer.create_indicator_map(
            regional_data.to_dataframe(), 'output_gap', 'Output Gap'
        )
        
        # Parameter estimation results
        visualizations['parameter_plots'] = self.parameter_visualizer.plot_parameter_estimates(
            regional_params
        )
        
        # Policy mistake decomposition
        visualizations['mistake_decomposition'] = self.policy_visualizer.plot_mistake_decomposition(
            policy_results['mistake_decomposition']
        )
        
        # Counterfactual comparisons
        visualizations['counterfactual_comparison'] = self.counterfactual_visualizer.plot_scenario_comparison(
            counterfactual_results['scenarios']
        )
        
        return visualizations
    
    def _generate_comprehensive_report(self, analysis_name: str, regional_data: Any,
                                     regional_params: Any, policy_results: Dict[str, Any],
                                     counterfactual_results: Dict[str, Any],
                                     visualizations: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""
        report_path = self.report_generator.generate_comprehensive_report(
            analysis_name=analysis_name,
            regional_data=regional_data,
            regional_parameters=regional_params,
            policy_analysis=policy_results,
            counterfactual_analysis=counterfactual_results,
            visualizations=visualizations,
            config=self.config
        )
        
        return report_path
    
    def run_quick_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run a quick analysis with default settings for testing purposes.
        
        Args:
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing basic analysis results
        """
        logger.info("Running quick analysis workflow")
        
        try:
            # Use subset of regions for speed
            regions = self.config.regions[:3] if len(self.config.regions) > 3 else self.config.regions
            
            # Retrieve data
            regional_data = self._retrieve_and_prepare_data(start_date, end_date, regions)
            
            # Basic parameter estimation
            spatial_weights, regional_params = self._estimate_parameters(regional_data)
            
            # Basic policy analysis
            optimal_policy = self.optimal_policy_calc.compute_optimal_policy_path(
                regional_data, regional_params
            )
            
            return {
                'data_shape': regional_data.to_dataframe().shape,
                'regions_analyzed': regions,
                'parameters_estimated': True,
                'optimal_policy_computed': True,
                'spatial_weights_shape': spatial_weights.shape
            }
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {e}")
            raise RegionalMonetaryPolicyError(f"Quick analysis failed: {e}")
    
    def validate_system_integration(self) -> Dict[str, bool]:
        """
        Validate that all system components are properly integrated.
        
        Returns:
            Dictionary with validation results for each component
        """
        logger.info("Validating system integration")
        
        validation_results = {}
        
        try:
            # Test data layer integration
            validation_results['fred_client'] = self.fred_client.validate_api_key()
            validation_results['data_manager'] = hasattr(self.data_manager, 'fred_client')
            
            # Test econometric layer integration
            validation_results['spatial_handler'] = len(self.spatial_handler.regions) > 0
            validation_results['parameter_estimator'] = hasattr(self.parameter_estimator, 'estimation_config')
            
            # Test policy layer integration
            validation_results['mistake_decomposer'] = hasattr(self.mistake_decomposer, 'decompose_policy_mistakes')
            validation_results['optimal_policy_calc'] = hasattr(self.optimal_policy_calc, 'compute_optimal_policy_path')
            validation_results['counterfactual_engine'] = hasattr(self.counterfactual_engine, 'generate_baseline_scenario')
            
            # Test presentation layer integration
            validation_results['map_visualizer'] = hasattr(self.map_visualizer, 'create_indicator_map')
            validation_results['timeseries_visualizer'] = hasattr(self.timeseries_visualizer, 'plot_policy_transmission')
            validation_results['parameter_visualizer'] = hasattr(self.parameter_visualizer, 'plot_parameter_estimates')
            validation_results['policy_visualizer'] = hasattr(self.policy_visualizer, 'plot_mistake_decomposition')
            validation_results['counterfactual_visualizer'] = hasattr(self.counterfactual_visualizer, 'plot_scenario_comparison')
            validation_results['report_generator'] = hasattr(self.report_generator, 'generate_comprehensive_report')
            
            # Overall integration status
            validation_results['overall_integration'] = all(validation_results.values())
            
            logger.info(f"System integration validation completed: {validation_results}")
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            validation_results['error'] = str(e)
            validation_results['overall_integration'] = False
            
        return validation_results
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and results."""
        return {
            'current_workflow': self.current_workflow,
            'completed_workflows': list(self.workflow_results.keys()),
            'system_status': self.validate_system_integration()
        }