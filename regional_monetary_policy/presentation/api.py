"""
Programmatic access API for custom analysis workflows.

This module provides a high-level API for accessing analysis results,
running custom analyses, and integrating the regional monetary policy
framework into external workflows.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from ..data.models import RegionalDataset, ValidationReport
from ..data.data_manager import DataManager
from ..data.fred_client import FREDClient
from ..econometric.models import RegionalParameters, EstimationResults
from ..econometric.parameter_estimator import ParameterEstimator
from ..econometric.spatial_handler import SpatialModelHandler
from ..policy.models import (
    PolicyScenario, PolicyMistakeComponents, 
    ComparisonResults, WelfareDecomposition
)
from ..policy.mistake_decomposer import PolicyMistakeDecomposer
from ..policy.optimal_policy import OptimalPolicyCalculator
from ..policy.counterfactual_engine import CounterfactualEngine
from ..config.settings import AnalysisSettings
from .report_generator import ReportGenerator, DataExporter, ChartExporter
from .visualizers import (
    RegionalMapVisualizer, TimeSeriesVisualizer, 
    ParameterVisualizer, PolicyAnalysisVisualizer, CounterfactualVisualizer
)


class RegionalMonetaryPolicyAPI:
    """
    High-level API for regional monetary policy analysis.
    
    Provides programmatic access to all analysis components with
    simplified interfaces for custom workflows and integration
    with external systems.
    """
    
    def __init__(
        self, 
        fred_api_key: str,
        cache_dir: str = "data/cache",
        output_dir: str = "output"
    ):
        """
        Initialize the API with required components.
        
        Args:
            fred_api_key: FRED API key for data access
            cache_dir: Directory for data caching
            output_dir: Directory for output files
        """
        # Settings will be initialized when needed
        self.settings = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.fred_client = FREDClient(fred_api_key, cache_dir)
        self.data_manager = DataManager(self.fred_client)
        
        # Initialize analysis components (will be set after data loading)
        self.spatial_handler = None
        self.parameter_estimator = None
        self.policy_decomposer = None
        self.optimal_calculator = None
        self.counterfactual_engine = None
        
        # Initialize presentation components
        self.report_generator = ReportGenerator(str(self.output_dir / "reports"))
        self.data_exporter = DataExporter(str(self.output_dir / "exports"))
        self.chart_exporter = ChartExporter(str(self.output_dir / "charts"))
        
        # Store analysis results
        self.regional_data = None
        self.regional_parameters = None
        self.policy_analysis = None
        self.counterfactual_results = None
        
    def load_regional_data(
        self,
        regions: List[str],
        start_date: str,
        end_date: str,
        indicators: Optional[List[str]] = None
    ) -> RegionalDataset:
        """
        Load regional economic data from FRED.
        
        Args:
            regions: List of region identifiers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            indicators: List of economic indicators to load
            
        Returns:
            Regional dataset with loaded data
        """
        if indicators is None:
            indicators = ['output_gap', 'inflation', 'interest_rate']
        
        self.regional_data = self.data_manager.load_regional_data(
            regions, indicators, start_date, end_date
        )
        
        return self.regional_data
    
    def estimate_parameters(
        self,
        spatial_weights_config: Optional[Dict[str, Any]] = None,
        estimation_config: Optional[Dict[str, Any]] = None
    ) -> RegionalParameters:
        """
        Estimate regional structural parameters.
        
        Args:
            spatial_weights_config: Configuration for spatial weight construction
            estimation_config: Configuration for parameter estimation
            
        Returns:
            Estimated regional parameters
        """
        if self.regional_data is None:
            raise ValueError("Regional data must be loaded first")
        
        # Initialize spatial handler
        regions = list(self.regional_data.output_gaps.index)
        self.spatial_handler = SpatialModelHandler(regions)
        
        # Construct spatial weights
        if spatial_weights_config:
            spatial_weights = self.spatial_handler.construct_weights(**spatial_weights_config)
        else:
            # Use default configuration
            spatial_weights = self.spatial_handler.construct_default_weights()
        
        # Initialize parameter estimator
        self.parameter_estimator = ParameterEstimator(
            spatial_weights, 
            estimation_config or {}
        )
        
        # Run estimation
        self.regional_parameters = self.parameter_estimator.estimate_parameters(
            self.regional_data
        )
        
        return self.regional_parameters
    
    def analyze_policy_mistakes(
        self,
        fed_policy_rates: pd.Series,
        real_time_data: Optional[RegionalDataset] = None
    ) -> PolicyMistakeComponents:
        """
        Analyze policy mistakes and decompose into components.
        
        Args:
            fed_policy_rates: Historical Fed policy rates
            real_time_data: Real-time data available to Fed (optional)
            
        Returns:
            Policy mistake decomposition results
        """
        if self.regional_parameters is None:
            raise ValueError("Parameters must be estimated first")
        
        # Initialize policy analysis components
        self.policy_decomposer = PolicyMistakeDecomposer(self.regional_parameters)
        self.optimal_calculator = OptimalPolicyCalculator(self.regional_parameters)
        
        # Use real-time data if provided, otherwise use full dataset
        analysis_data = real_time_data or self.regional_data
        
        # Compute optimal policy rates
        optimal_rates = []
        for date in fed_policy_rates.index:
            if date in analysis_data.output_gaps.columns:
                regional_conditions = pd.DataFrame({
                    'output_gap': analysis_data.output_gaps[date],
                    'inflation': analysis_data.inflation_rates[date]
                })
                optimal_rate = self.optimal_calculator.compute_optimal_rate(regional_conditions)
                optimal_rates.append(optimal_rate)
            else:
                optimal_rates.append(np.nan)
        
        optimal_rates_series = pd.Series(optimal_rates, index=fed_policy_rates.index)
        
        # Decompose policy mistakes
        self.policy_analysis = self.policy_decomposer.decompose_policy_mistakes(
            fed_policy_rates, optimal_rates_series, analysis_data
        )
        
        return self.policy_analysis
    
    def run_counterfactual_analysis(
        self,
        scenarios: Optional[List[str]] = None
    ) -> List[PolicyScenario]:
        """
        Run counterfactual policy analysis.
        
        Args:
            scenarios: List of scenario names to analyze
            
        Returns:
            List of policy scenarios with results
        """
        if self.regional_parameters is None:
            raise ValueError("Parameters must be estimated first")
        
        # Initialize counterfactual engine
        self.counterfactual_engine = CounterfactualEngine(self.regional_parameters)
        
        # Default scenarios if none specified
        if scenarios is None:
            scenarios = ['baseline', 'perfect_info', 'optimal_regional', 'perfect_regional']
        
        # Generate scenarios
        self.counterfactual_results = []
        
        for scenario_name in scenarios:
            if scenario_name == 'baseline':
                scenario = self.counterfactual_engine.generate_baseline_scenario(self.regional_data)
            elif scenario_name == 'perfect_info':
                scenario = self.counterfactual_engine.generate_perfect_info_scenario(self.regional_data)
            elif scenario_name == 'optimal_regional':
                scenario = self.counterfactual_engine.generate_optimal_regional_scenario(self.regional_data)
            elif scenario_name == 'perfect_regional':
                scenario = self.counterfactual_engine.generate_perfect_regional_scenario(self.regional_data)
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            
            self.counterfactual_results.append(scenario)
        
        return self.counterfactual_results
    
    def create_visualizations(
        self,
        visualization_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create standard visualizations for analysis results.
        
        Args:
            visualization_types: List of visualization types to create
            
        Returns:
            Dictionary mapping visualization names to Plotly figures
        """
        if visualization_types is None:
            visualization_types = [
                'regional_map', 'parameter_estimates', 'policy_transmission',
                'mistake_decomposition', 'counterfactual_comparison'
            ]
        
        figures = {}
        
        # Regional map visualizations
        if 'regional_map' in visualization_types and self.regional_data is not None:
            regions = list(self.regional_data.output_gaps.index)
            map_viz = RegionalMapVisualizer(regions)
            
            # Create multi-indicator map
            indicators = {
                'Average Output Gap': self.regional_data.output_gaps.mean(axis=1),
                'Average Inflation': self.regional_data.inflation_rates.mean(axis=1),
                'Output Volatility': self.regional_data.output_gaps.std(axis=1),
                'Inflation Volatility': self.regional_data.inflation_rates.std(axis=1)
            }
            figures['regional_indicators_map'] = map_viz.create_multi_indicator_map(indicators)
        
        # Parameter estimation visualizations
        if 'parameter_estimates' in visualization_types and self.regional_parameters is not None:
            param_viz = ParameterVisualizer()
            figures['parameter_estimates'] = param_viz.create_parameter_estimates_plot(self.regional_parameters)
            figures['parameter_table'] = param_viz.create_parameter_comparison_table(self.regional_parameters)
        
        # Policy transmission visualizations
        if 'policy_transmission' in visualization_types and self.regional_data is not None:
            ts_viz = TimeSeriesVisualizer()
            
            # Create policy transmission plot
            policy_rates = self.regional_data.interest_rates
            regional_outcomes = pd.concat([
                self.regional_data.output_gaps,
                self.regional_data.inflation_rates
            ])
            
            figures['policy_transmission'] = ts_viz.create_policy_transmission_plot(
                policy_rates, regional_outcomes
            )
        
        # Policy mistake decomposition visualizations
        if 'mistake_decomposition' in visualization_types and self.policy_analysis is not None:
            policy_viz = PolicyAnalysisVisualizer()
            figures['mistake_decomposition'] = policy_viz.create_mistake_decomposition_plot(
                self.policy_analysis
            )
        
        # Counterfactual comparison visualizations
        if 'counterfactual_comparison' in visualization_types and self.counterfactual_results is not None:
            cf_viz = CounterfactualVisualizer()
            figures['counterfactual_comparison'] = cf_viz.create_scenario_comparison_plot(
                self.counterfactual_results
            )
        
        return figures
    
    def export_results(
        self,
        export_formats: Optional[List[str]] = None,
        include_charts: bool = True
    ) -> Dict[str, Any]:
        """
        Export all analysis results in specified formats.
        
        Args:
            export_formats: List of export formats ('csv', 'json', 'latex')
            include_charts: Whether to export charts as well
            
        Returns:
            Dictionary with paths to exported files
        """
        if export_formats is None:
            export_formats = ['csv', 'json', 'latex']
        
        exported_files = {}
        
        # Export regional data
        if self.regional_data is not None:
            exported_files['regional_data'] = self.data_exporter.export_regional_data(
                self.regional_data, formats=export_formats
            )
        
        # Export parameter estimates
        if self.regional_parameters is not None:
            exported_files['parameters'] = self.data_exporter.export_parameter_estimates(
                self.regional_parameters, formats=export_formats
            )
        
        # Export policy analysis
        if self.policy_analysis is not None:
            exported_files['policy_analysis'] = self.data_exporter.export_policy_analysis(
                self.policy_analysis, formats=export_formats
            )
        
        # Export counterfactual results
        if self.counterfactual_results is not None:
            # Create comparison results
            comparison_results = ComparisonResults(self.counterfactual_results)
            exported_files['counterfactual'] = self.data_exporter.export_counterfactual_results(
                self.counterfactual_results, comparison_results, formats=export_formats
            )
        
        # Export charts if requested
        if include_charts:
            figures = self.create_visualizations()
            exported_files['charts'] = self.chart_exporter.export_multiple_figures(
                figures, formats=['png', 'pdf']
            )
        
        return exported_files
    
    def generate_reports(
        self,
        report_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive analysis reports.
        
        Args:
            report_types: List of report types to generate
            
        Returns:
            Dictionary mapping report types to file paths
        """
        if report_types is None:
            report_types = ['comprehensive', 'methodology', 'executive_summary']
        
        generated_reports = {}
        
        # Comprehensive report
        if 'comprehensive' in report_types:
            if all([
                self.regional_data is not None,
                self.regional_parameters is not None,
                self.policy_analysis is not None,
                self.counterfactual_results is not None
            ]):
                report_path = self.report_generator.generate_comprehensive_report(
                    self.regional_data,
                    self.regional_parameters,
                    self.policy_analysis,
                    self.counterfactual_results
                )
                generated_reports['comprehensive'] = report_path
        
        # Methodology report
        if 'methodology' in report_types:
            estimation_config = getattr(self.parameter_estimator, 'config', {}) if self.parameter_estimator else {}
            model_spec = {
                'regions': list(self.regional_data.output_gaps.index) if self.regional_data else [],
                'time_period': {
                    'start': str(self.regional_data.output_gaps.columns[0]) if self.regional_data else None,
                    'end': str(self.regional_data.output_gaps.columns[-1]) if self.regional_data else None
                }
            }
            
            methodology_path = self.report_generator.generate_methodology_report(
                estimation_config, model_spec
            )
            generated_reports['methodology'] = methodology_path
        
        # Executive summary
        if 'executive_summary' in report_types:
            if self.counterfactual_results is not None:
                key_findings = self._extract_key_findings()
                policy_implications = self._extract_policy_implications()
                welfare_gains = {
                    scenario.name: scenario.welfare_outcome 
                    for scenario in self.counterfactual_results
                }
                
                summary_path = self.report_generator.generate_executive_summary(
                    key_findings, policy_implications, welfare_gains
                )
                generated_reports['executive_summary'] = summary_path
        
        return generated_reports
    
    def get_analysis_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata about the analysis.
        
        Returns:
            Dictionary with detailed metadata
        """
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_info': {},
            'estimation_info': {},
            'analysis_info': {}
        }
        
        # Data information
        if self.regional_data is not None:
            metadata['data_info'] = {
                'n_regions': len(self.regional_data.output_gaps.index),
                'regions': list(self.regional_data.output_gaps.index),
                'time_period': {
                    'start': str(self.regional_data.output_gaps.columns[0]),
                    'end': str(self.regional_data.output_gaps.columns[-1]),
                    'n_periods': len(self.regional_data.output_gaps.columns)
                },
                'data_sources': self.regional_data.metadata.get('sources', {}),
                'data_quality': {
                    'output_gaps_missing': int(self.regional_data.output_gaps.isnull().sum().sum()),
                    'inflation_missing': int(self.regional_data.inflation_rates.isnull().sum().sum())
                }
            }
        
        # Estimation information
        if self.regional_parameters is not None:
            metadata['estimation_info'] = {
                'parameter_ranges': {
                    'sigma': {
                        'min': float(self.regional_parameters.sigma.min()),
                        'max': float(self.regional_parameters.sigma.max()),
                        'mean': float(self.regional_parameters.sigma.mean())
                    },
                    'kappa': {
                        'min': float(self.regional_parameters.kappa.min()),
                        'max': float(self.regional_parameters.kappa.max()),
                        'mean': float(self.regional_parameters.kappa.mean())
                    }
                },
                'heterogeneity_measures': {
                    'sigma_cv': float(self.regional_parameters.sigma.std() / self.regional_parameters.sigma.mean()),
                    'kappa_cv': float(self.regional_parameters.kappa.std() / self.regional_parameters.kappa.mean())
                }
            }
        
        # Analysis information
        if self.policy_analysis is not None:
            metadata['analysis_info']['policy_mistakes'] = {
                'total_mistake': float(self.policy_analysis.total_mistake),
                'components': {
                    'information_effect': float(self.policy_analysis.information_effect),
                    'weight_misallocation': float(self.policy_analysis.weight_misallocation_effect),
                    'parameter_misspec': float(self.policy_analysis.parameter_misspecification_effect),
                    'inflation_response': float(self.policy_analysis.inflation_response_effect)
                }
            }
        
        if self.counterfactual_results is not None:
            metadata['analysis_info']['counterfactual'] = {
                'n_scenarios': len(self.counterfactual_results),
                'scenarios': [scenario.name for scenario in self.counterfactual_results],
                'welfare_outcomes': {
                    scenario.name: float(scenario.welfare_outcome) 
                    for scenario in self.counterfactual_results
                }
            }
        
        return metadata
    
    def save_analysis_state(self, filepath: str) -> None:
        """
        Save the current analysis state to file.
        
        Args:
            filepath: Path to save the analysis state
        """
        state = {
            'metadata': self.get_analysis_metadata(),
            'has_data': self.regional_data is not None,
            'has_parameters': self.regional_parameters is not None,
            'has_policy_analysis': self.policy_analysis is not None,
            'has_counterfactual': self.counterfactual_results is not None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def run_custom_analysis(
        self,
        analysis_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Run a custom analysis function with access to all components.
        
        Args:
            analysis_function: Custom analysis function to run
            *args: Positional arguments for the analysis function
            **kwargs: Keyword arguments for the analysis function
            
        Returns:
            Result of the custom analysis function
        """
        # Provide access to all components
        api_components = {
            'regional_data': self.regional_data,
            'regional_parameters': self.regional_parameters,
            'policy_analysis': self.policy_analysis,
            'counterfactual_results': self.counterfactual_results,
            'spatial_handler': self.spatial_handler,
            'parameter_estimator': self.parameter_estimator,
            'policy_decomposer': self.policy_decomposer,
            'optimal_calculator': self.optimal_calculator,
            'counterfactual_engine': self.counterfactual_engine,
            'data_manager': self.data_manager,
            'fred_client': self.fred_client
        }
        
        return analysis_function(api_components, *args, **kwargs)
    
    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from analysis results."""
        findings = {}
        
        if self.regional_parameters is not None:
            findings['parameter_heterogeneity'] = {
                'sigma_heterogeneity': float(self.regional_parameters.sigma.std()),
                'kappa_heterogeneity': float(self.regional_parameters.kappa.std())
            }
        
        if self.policy_analysis is not None:
            findings['policy_mistakes'] = {
                'dominant_component': self._find_dominant_mistake_component(),
                'total_magnitude': float(abs(self.policy_analysis.total_mistake))
            }
        
        return findings
    
    def _extract_policy_implications(self) -> List[str]:
        """Extract policy implications from analysis results."""
        implications = []
        
        if self.regional_parameters is not None:
            # Check for significant heterogeneity
            sigma_cv = self.regional_parameters.sigma.std() / self.regional_parameters.sigma.mean()
            if sigma_cv > 0.2:  # Threshold for significant heterogeneity
                implications.append(
                    "Significant regional heterogeneity in interest rate sensitivity "
                    "suggests the need for region-specific policy considerations."
                )
        
        if self.policy_analysis is not None:
            # Check dominant mistake component
            dominant = self._find_dominant_mistake_component()
            if dominant == 'Information Effect':
                implications.append(
                    "Information limitations are the primary source of policy mistakes, "
                    "suggesting benefits from improved real-time data collection."
                )
            elif dominant == 'Weight Misallocation':
                implications.append(
                    "Regional weight misallocation is the main policy issue, "
                    "indicating need for better regional representation in policy decisions."
                )
        
        if self.counterfactual_results is not None:
            # Check welfare gains
            welfare_outcomes = [scenario.welfare_outcome for scenario in self.counterfactual_results]
            if max(welfare_outcomes) - min(welfare_outcomes) > 0.001:  # Significant welfare differences
                implications.append(
                    "Substantial welfare gains are possible through improved monetary policy approaches."
                )
        
        return implications
    
    def _find_dominant_mistake_component(self) -> str:
        """Find the dominant component in policy mistake decomposition."""
        if self.policy_analysis is None:
            return "Unknown"
        
        components = {
            'Information Effect': abs(self.policy_analysis.information_effect),
            'Weight Misallocation': abs(self.policy_analysis.weight_misallocation_effect),
            'Parameter Misspecification': abs(self.policy_analysis.parameter_misspecification_effect),
            'Inflation Response': abs(self.policy_analysis.inflation_response_effect)
        }
        
        return max(components, key=components.get)


class AnalysisWorkflow:
    """
    Pre-defined analysis workflows for common research questions.
    
    Provides standardized workflows that combine multiple analysis
    steps for typical regional monetary policy research questions.
    """
    
    def __init__(self, api: RegionalMonetaryPolicyAPI):
        """
        Initialize workflow with API instance.
        
        Args:
            api: Regional monetary policy API instance
        """
        self.api = api
    
    def full_analysis_workflow(
        self,
        regions: List[str],
        start_date: str,
        end_date: str,
        fed_policy_rates: pd.Series,
        export_results: bool = True,
        generate_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete analysis workflow from data loading to reporting.
        
        Args:
            regions: List of region identifiers
            start_date: Analysis start date
            end_date: Analysis end date
            fed_policy_rates: Historical Fed policy rates
            export_results: Whether to export results
            generate_reports: Whether to generate reports
            
        Returns:
            Dictionary with all analysis results and file paths
        """
        results = {}
        
        # Step 1: Load data
        print("Loading regional data...")
        regional_data = self.api.load_regional_data(regions, start_date, end_date)
        results['regional_data'] = regional_data
        
        # Step 2: Estimate parameters
        print("Estimating parameters...")
        regional_parameters = self.api.estimate_parameters()
        results['regional_parameters'] = regional_parameters
        
        # Step 3: Analyze policy mistakes
        print("Analyzing policy mistakes...")
        policy_analysis = self.api.analyze_policy_mistakes(fed_policy_rates)
        results['policy_analysis'] = policy_analysis
        
        # Step 4: Run counterfactual analysis
        print("Running counterfactual analysis...")
        counterfactual_results = self.api.run_counterfactual_analysis()
        results['counterfactual_results'] = counterfactual_results
        
        # Step 5: Create visualizations
        print("Creating visualizations...")
        figures = self.api.create_visualizations()
        results['figures'] = figures
        
        # Step 6: Export results (optional)
        if export_results:
            print("Exporting results...")
            exported_files = self.api.export_results()
            results['exported_files'] = exported_files
        
        # Step 7: Generate reports (optional)
        if generate_reports:
            print("Generating reports...")
            reports = self.api.generate_reports()
            results['reports'] = reports
        
        # Step 8: Get metadata
        metadata = self.api.get_analysis_metadata()
        results['metadata'] = metadata
        
        print("Analysis workflow completed successfully!")
        return results
    
    def parameter_sensitivity_analysis(
        self,
        parameter_variations: Dict[str, List[float]],
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run parameter sensitivity analysis.
        
        Args:
            parameter_variations: Dictionary of parameters and their variation ranges
            base_config: Base configuration for estimation
            
        Returns:
            Sensitivity analysis results
        """
        if self.api.regional_data is None:
            raise ValueError("Regional data must be loaded first")
        
        sensitivity_results = {}
        base_config = base_config or {}
        
        for param_name, param_values in parameter_variations.items():
            param_results = []
            
            for param_value in param_values:
                # Create modified configuration
                modified_config = base_config.copy()
                modified_config[param_name] = param_value
                
                # Re-estimate with modified configuration
                try:
                    regional_params = self.api.estimate_parameters(
                        estimation_config=modified_config
                    )
                    
                    # Store key results
                    param_results.append({
                        'parameter_value': param_value,
                        'sigma_mean': float(regional_params.sigma.mean()),
                        'kappa_mean': float(regional_params.kappa.mean()),
                        'sigma_std': float(regional_params.sigma.std()),
                        'kappa_std': float(regional_params.kappa.std())
                    })
                    
                except Exception as e:
                    param_results.append({
                        'parameter_value': param_value,
                        'error': str(e)
                    })
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def robustness_check_workflow(
        self,
        alternative_specifications: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run robustness checks with alternative model specifications.
        
        Args:
            alternative_specifications: List of alternative estimation configurations
            
        Returns:
            Robustness check results
        """
        if self.api.regional_data is None:
            raise ValueError("Regional data must be loaded first")
        
        robustness_results = {}
        
        for i, spec in enumerate(alternative_specifications):
            spec_name = spec.get('name', f'Specification_{i+1}')
            
            try:
                # Estimate with alternative specification
                regional_params = self.api.estimate_parameters(
                    spatial_weights_config=spec.get('spatial_config'),
                    estimation_config=spec.get('estimation_config')
                )
                
                # Run policy analysis if Fed rates are available
                if hasattr(self.api, '_fed_policy_rates'):
                    policy_analysis = self.api.analyze_policy_mistakes(
                        self.api._fed_policy_rates
                    )
                else:
                    policy_analysis = None
                
                # Store results
                robustness_results[spec_name] = {
                    'regional_parameters': regional_params,
                    'policy_analysis': policy_analysis,
                    'specification': spec
                }
                
            except Exception as e:
                robustness_results[spec_name] = {
                    'error': str(e),
                    'specification': spec
                }
        
        return robustness_results