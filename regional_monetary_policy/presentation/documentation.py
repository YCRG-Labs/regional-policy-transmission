"""
Documentation and metadata generation for regional monetary policy analysis.

This module provides comprehensive documentation generation capabilities,
including detailed metadata about data sources, estimation procedures,
and model assumptions for reproducible research.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import asdict

from ..data.models import RegionalDataset, ValidationReport
from ..econometric.models import RegionalParameters, EstimationResults
from ..policy.models import (
    PolicyScenario, PolicyMistakeComponents, 
    ComparisonResults, WelfareDecomposition
)
from ..config.settings import AnalysisSettings


class MetadataGenerator:
    """
    Generates comprehensive metadata for analysis results.
    
    Creates detailed documentation about data sources, estimation
    procedures, model assumptions, and analysis configurations
    for reproducible research.
    """
    
    def __init__(self, output_dir: str = "metadata"):
        """
        Initialize metadata generator.
        
        Args:
            output_dir: Directory for metadata files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_data_metadata(
        self,
        regional_data: RegionalDataset,
        data_sources: Optional[Dict[str, Any]] = None,
        collection_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for regional dataset.
        
        Args:
            regional_data: Regional economic dataset
            data_sources: Information about data sources
            collection_info: Information about data collection process
            
        Returns:
            Comprehensive data metadata dictionary
        """
        metadata = {
            'dataset_info': {
                'creation_timestamp': datetime.now().isoformat(),
                'dataset_type': 'regional_economic_data',
                'n_regions': len(regional_data.output_gaps.index),
                'regions': list(regional_data.output_gaps.index),
                'time_coverage': {
                    'start_date': str(regional_data.output_gaps.columns[0]),
                    'end_date': str(regional_data.output_gaps.columns[-1]),
                    'n_periods': len(regional_data.output_gaps.columns),
                    'frequency': self._infer_frequency(regional_data.output_gaps.columns)
                }
            },
            'variables': {
                'output_gaps': {
                    'description': 'Regional output gaps (deviation from potential output)',
                    'units': 'percentage points',
                    'source': 'FRED API',
                    'construction_method': 'HP filter or real-time estimates',
                    'missing_values': int(regional_data.output_gaps.isnull().sum().sum()),
                    'summary_statistics': self._compute_variable_summary(regional_data.output_gaps)
                },
                'inflation_rates': {
                    'description': 'Regional inflation rates (year-over-year)',
                    'units': 'percentage points (annualized)',
                    'source': 'FRED API',
                    'construction_method': 'CPI-based regional price indices',
                    'missing_values': int(regional_data.inflation_rates.isnull().sum().sum()),
                    'summary_statistics': self._compute_variable_summary(regional_data.inflation_rates)
                },
                'interest_rates': {
                    'description': 'Federal funds rate (policy interest rate)',
                    'units': 'percentage points (annualized)',
                    'source': 'FRED API',
                    'construction_method': 'Effective federal funds rate',
                    'missing_values': int(regional_data.interest_rates.isnull().sum()),
                    'summary_statistics': self._compute_series_summary(regional_data.interest_rates)
                }
            },
            'data_quality': {
                'completeness': {
                    'output_gaps': float(1 - regional_data.output_gaps.isnull().sum().sum() / regional_data.output_gaps.size),
                    'inflation_rates': float(1 - regional_data.inflation_rates.isnull().sum().sum() / regional_data.inflation_rates.size),
                    'interest_rates': float(1 - regional_data.interest_rates.isnull().sum() / len(regional_data.interest_rates))
                },
                'outliers': self._detect_outliers(regional_data),
                'structural_breaks': self._detect_structural_breaks(regional_data)
            },
            'data_sources': data_sources or self._get_default_data_sources(),
            'collection_info': collection_info or self._get_default_collection_info(),
            'vintage_information': self._extract_vintage_info(regional_data)
        }
        
        return metadata
    
    def generate_estimation_metadata(
        self,
        regional_params: RegionalParameters,
        estimation_config: Dict[str, Any],
        spatial_weights_info: Optional[Dict[str, Any]] = None,
        convergence_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata for parameter estimation results.
        
        Args:
            regional_params: Estimated regional parameters
            estimation_config: Configuration used for estimation
            spatial_weights_info: Information about spatial weight construction
            convergence_info: Information about estimation convergence
            
        Returns:
            Comprehensive estimation metadata dictionary
        """
        metadata = {
            'estimation_info': {
                'estimation_timestamp': datetime.now().isoformat(),
                'estimation_method': 'Three-stage GMM with spatial weights',
                'n_regions': regional_params.n_regions,
                'estimation_config': estimation_config,
                'convergence_achieved': convergence_info.get('converged', True) if convergence_info else True,
                'n_iterations': convergence_info.get('n_iterations', 'unknown') if convergence_info else 'unknown'
            },
            'model_specification': {
                'theoretical_framework': 'Multi-region New Keynesian DSGE with spatial spillovers',
                'key_equations': [
                    'Regional IS curve with spatial spillovers',
                    'Regional Phillips curve with spatial spillovers',
                    'Monetary policy reaction function'
                ],
                'identification_strategy': 'Regional variation and spatial exclusion restrictions',
                'spatial_structure': spatial_weights_info or self._get_default_spatial_info()
            },
            'parameter_estimates': {
                'sigma': {
                    'description': 'Interest rate sensitivity (inverse of intertemporal elasticity)',
                    'estimates': regional_params.sigma.tolist(),
                    'standard_errors': regional_params.standard_errors.get('sigma', []).tolist(),
                    'confidence_intervals': self._format_confidence_intervals(regional_params, 'sigma'),
                    'summary_statistics': {
                        'mean': float(regional_params.sigma.mean()),
                        'std': float(regional_params.sigma.std()),
                        'min': float(regional_params.sigma.min()),
                        'max': float(regional_params.sigma.max()),
                        'coefficient_of_variation': float(regional_params.sigma.std() / regional_params.sigma.mean())
                    }
                },
                'kappa': {
                    'description': 'Phillips curve slope (price stickiness parameter)',
                    'estimates': regional_params.kappa.tolist(),
                    'standard_errors': regional_params.standard_errors.get('kappa', []).tolist(),
                    'confidence_intervals': self._format_confidence_intervals(regional_params, 'kappa'),
                    'summary_statistics': {
                        'mean': float(regional_params.kappa.mean()),
                        'std': float(regional_params.kappa.std()),
                        'min': float(regional_params.kappa.min()),
                        'max': float(regional_params.kappa.max()),
                        'coefficient_of_variation': float(regional_params.kappa.std() / regional_params.kappa.mean())
                    }
                },
                'psi': {
                    'description': 'Demand spillover parameter',
                    'estimates': regional_params.psi.tolist(),
                    'standard_errors': regional_params.standard_errors.get('psi', []).tolist(),
                    'confidence_intervals': self._format_confidence_intervals(regional_params, 'psi'),
                    'summary_statistics': {
                        'mean': float(regional_params.psi.mean()),
                        'std': float(regional_params.psi.std()),
                        'min': float(regional_params.psi.min()),
                        'max': float(regional_params.psi.max())
                    }
                },
                'phi': {
                    'description': 'Price spillover parameter',
                    'estimates': regional_params.phi.tolist(),
                    'standard_errors': regional_params.standard_errors.get('phi', []).tolist(),
                    'confidence_intervals': self._format_confidence_intervals(regional_params, 'phi'),
                    'summary_statistics': {
                        'mean': float(regional_params.phi.mean()),
                        'std': float(regional_params.phi.std()),
                        'min': float(regional_params.phi.min()),
                        'max': float(regional_params.phi.max())
                    }
                },
                'beta': {
                    'description': 'Discount factor',
                    'estimates': regional_params.beta.tolist(),
                    'standard_errors': regional_params.standard_errors.get('beta', []).tolist(),
                    'confidence_intervals': self._format_confidence_intervals(regional_params, 'beta'),
                    'summary_statistics': {
                        'mean': float(regional_params.beta.mean()),
                        'std': float(regional_params.beta.std()),
                        'min': float(regional_params.beta.min()),
                        'max': float(regional_params.beta.max())
                    }
                }
            },
            'diagnostic_tests': self._generate_diagnostic_metadata(regional_params),
            'robustness_checks': self._generate_robustness_metadata()
        }
        
        return metadata
    
    def generate_policy_analysis_metadata(
        self,
        policy_analysis: PolicyMistakeComponents,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata for policy analysis results.
        
        Args:
            policy_analysis: Policy mistake decomposition results
            analysis_config: Configuration used for policy analysis
            
        Returns:
            Policy analysis metadata dictionary
        """
        metadata = {
            'analysis_info': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'Policy mistake decomposition (Theorem 4)',
                'theoretical_basis': 'Welfare-maximizing monetary policy with regional heterogeneity',
                'decomposition_method': 'Four-component mistake decomposition',
                'analysis_config': analysis_config or {}
            },
            'mistake_components': {
                'total_mistake': {
                    'value': float(policy_analysis.total_mistake),
                    'description': 'Total policy mistake (actual - optimal policy rate)',
                    'units': 'percentage points',
                    'interpretation': 'Positive values indicate overly tight policy, negative values indicate overly loose policy'
                },
                'information_effect': {
                    'value': float(policy_analysis.information_effect),
                    'description': 'Effect of imperfect information about regional conditions',
                    'units': 'percentage points',
                    'interpretation': 'Contribution of real-time data limitations to policy mistakes'
                },
                'weight_misallocation_effect': {
                    'value': float(policy_analysis.weight_misallocation_effect),
                    'description': 'Effect of suboptimal regional weights in policy decisions',
                    'units': 'percentage points',
                    'interpretation': 'Contribution of regional weight misallocation to policy mistakes'
                },
                'parameter_misspecification_effect': {
                    'value': float(policy_analysis.parameter_misspecification_effect),
                    'description': 'Effect of parameter misspecification in Fed model',
                    'units': 'percentage points',
                    'interpretation': 'Contribution of incorrect parameter assumptions to policy mistakes'
                },
                'inflation_response_effect': {
                    'value': float(policy_analysis.inflation_response_effect),
                    'description': 'Effect of suboptimal inflation response coefficient',
                    'units': 'percentage points',
                    'interpretation': 'Contribution of incorrect inflation response to policy mistakes'
                }
            },
            'relative_contributions': self._compute_relative_contributions(policy_analysis),
            'economic_interpretation': self._generate_economic_interpretation(policy_analysis),
            'policy_implications': self._generate_policy_implications(policy_analysis)
        }
        
        return metadata
    
    def generate_counterfactual_metadata(
        self,
        scenarios: List[PolicyScenario],
        comparison_results: ComparisonResults
    ) -> Dict[str, Any]:
        """
        Generate metadata for counterfactual analysis results.
        
        Args:
            scenarios: List of policy scenarios
            comparison_results: Scenario comparison results
            
        Returns:
            Counterfactual analysis metadata dictionary
        """
        metadata = {
            'analysis_info': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': 'Counterfactual policy analysis',
                'n_scenarios': len(scenarios),
                'welfare_function': 'Regional social welfare with heterogeneous preferences',
                'comparison_method': 'Welfare ranking and decomposition'
            },
            'scenarios': {},
            'welfare_comparison': {
                'welfare_outcomes': {},
                'welfare_ranking': [],
                'welfare_gains': {},
                'ranking_verification': self._verify_welfare_ranking(scenarios)
            },
            'economic_interpretation': self._generate_counterfactual_interpretation(scenarios),
            'policy_recommendations': self._generate_policy_recommendations(scenarios)
        }
        
        # Add individual scenario metadata
        for scenario in scenarios:
            scenario_metadata = {
                'name': scenario.name,
                'scenario_type': scenario.scenario_type,
                'description': self._get_scenario_description(scenario.scenario_type),
                'welfare_outcome': float(scenario.welfare_outcome),
                'policy_statistics': {
                    'mean_rate': float(scenario.policy_rates.mean()),
                    'std_rate': float(scenario.policy_rates.std()),
                    'min_rate': float(scenario.policy_rates.min()),
                    'max_rate': float(scenario.policy_rates.max())
                }
            }
            
            if hasattr(scenario, 'regional_outcomes') and scenario.regional_outcomes is not None:
                scenario_metadata['regional_impacts'] = self._compute_regional_impacts(scenario)
            
            metadata['scenarios'][scenario.name] = scenario_metadata
            metadata['welfare_comparison']['welfare_outcomes'][scenario.name] = float(scenario.welfare_outcome)
        
        # Compute welfare ranking
        welfare_sorted = sorted(scenarios, key=lambda x: x.welfare_outcome, reverse=True)
        metadata['welfare_comparison']['welfare_ranking'] = [s.name for s in welfare_sorted]
        
        # Compute welfare gains relative to baseline
        baseline_welfare = next((s.welfare_outcome for s in scenarios if 'baseline' in s.name.lower()), 0)
        for scenario in scenarios:
            metadata['welfare_comparison']['welfare_gains'][scenario.name] = float(
                scenario.welfare_outcome - baseline_welfare
            )
        
        return metadata
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str,
        format: str = 'json'
    ) -> str:
        """
        Save metadata to file in specified format.
        
        Args:
            metadata: Metadata dictionary to save
            filename: Output filename (without extension)
            format: Output format ('json' or 'yaml')
            
        Returns:
            Path to saved metadata file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filepath = self.output_dir / f"{filename}_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        elif format == 'yaml':
            filepath = self.output_dir / f"{filename}_{timestamp}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def generate_complete_metadata(
        self,
        regional_data: RegionalDataset,
        regional_params: RegionalParameters,
        policy_analysis: PolicyMistakeComponents,
        counterfactual_results: List[PolicyScenario],
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete metadata for entire analysis.
        
        Args:
            regional_data: Regional economic dataset
            regional_params: Parameter estimation results
            policy_analysis: Policy mistake decomposition
            counterfactual_results: Counterfactual scenario results
            analysis_config: Overall analysis configuration
            
        Returns:
            Complete metadata dictionary
        """
        complete_metadata = {
            'analysis_overview': {
                'creation_timestamp': datetime.now().isoformat(),
                'analysis_type': 'Regional Monetary Policy Analysis',
                'framework': 'Multi-region New Keynesian DSGE with spatial spillovers',
                'software_version': '1.0.0',
                'analysis_config': analysis_config or {}
            },
            'data_metadata': self.generate_data_metadata(regional_data),
            'estimation_metadata': self.generate_estimation_metadata(
                regional_params, 
                analysis_config.get('estimation', {}) if analysis_config else {}
            ),
            'policy_analysis_metadata': self.generate_policy_analysis_metadata(
                policy_analysis,
                analysis_config.get('policy_analysis', {}) if analysis_config else {}
            ),
            'counterfactual_metadata': self.generate_counterfactual_metadata(
                counterfactual_results,
                ComparisonResults(counterfactual_results)
            ),
            'reproducibility_info': self._generate_reproducibility_info(analysis_config)
        }
        
        return complete_metadata
    
    def _infer_frequency(self, date_index) -> str:
        """Infer the frequency of the time series."""
        if len(date_index) < 2:
            return 'unknown'
        
        # Convert to pandas datetime if not already
        if not isinstance(date_index, pd.DatetimeIndex):
            date_index = pd.to_datetime(date_index)
        
        # Compute typical difference
        diff = date_index[1] - date_index[0]
        
        if diff.days <= 1:
            return 'daily'
        elif diff.days <= 7:
            return 'weekly'
        elif diff.days <= 31:
            return 'monthly'
        elif diff.days <= 92:
            return 'quarterly'
        else:
            return 'annual'
    
    def _compute_variable_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for a variable across regions and time."""
        return {
            'overall_mean': float(data.mean().mean()),
            'overall_std': float(data.std().std()),
            'regional_means': data.mean(axis=1).to_dict(),
            'temporal_means': data.mean(axis=0).to_dict(),
            'cross_regional_correlation': float(data.T.corr().mean().mean()),
            'temporal_persistence': float(data.apply(lambda x: x.autocorr(lag=1), axis=1).mean())
        }
    
    def _compute_series_summary(self, data: pd.Series) -> Dict[str, Any]:
        """Compute summary statistics for a time series."""
        return {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'autocorrelation_lag1': float(data.autocorr(lag=1)),
            'trend': 'increasing' if data.iloc[-1] > data.iloc[0] else 'decreasing'
        }
    
    def _detect_outliers(self, regional_data: RegionalDataset) -> Dict[str, Any]:
        """Detect outliers in the regional dataset."""
        outliers = {}
        
        # Output gaps outliers
        output_outliers = []
        for region in regional_data.output_gaps.index:
            region_data = regional_data.output_gaps.loc[region]
            q1, q3 = region_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (region_data < q1 - 1.5*iqr) | (region_data > q3 + 1.5*iqr)
            if outlier_mask.any():
                output_outliers.append({
                    'region': region,
                    'n_outliers': int(outlier_mask.sum()),
                    'outlier_dates': region_data[outlier_mask].index.tolist()
                })
        
        outliers['output_gaps'] = output_outliers
        
        # Similar for inflation rates
        inflation_outliers = []
        for region in regional_data.inflation_rates.index:
            region_data = regional_data.inflation_rates.loc[region]
            q1, q3 = region_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (region_data < q1 - 1.5*iqr) | (region_data > q3 + 1.5*iqr)
            if outlier_mask.any():
                inflation_outliers.append({
                    'region': region,
                    'n_outliers': int(outlier_mask.sum()),
                    'outlier_dates': region_data[outlier_mask].index.tolist()
                })
        
        outliers['inflation_rates'] = inflation_outliers
        
        return outliers
    
    def _detect_structural_breaks(self, regional_data: RegionalDataset) -> Dict[str, Any]:
        """Detect potential structural breaks in the data."""
        # Simplified structural break detection
        breaks = {
            'output_gaps': [],
            'inflation_rates': [],
            'method': 'Simple variance change detection'
        }
        
        # Check for variance changes in output gaps
        for region in regional_data.output_gaps.index:
            region_data = regional_data.output_gaps.loc[region].dropna()
            if len(region_data) > 20:  # Need sufficient data
                mid_point = len(region_data) // 2
                first_half_var = region_data.iloc[:mid_point].var()
                second_half_var = region_data.iloc[mid_point:].var()
                
                # Simple test: if variance changes by more than 50%
                if abs(first_half_var - second_half_var) / first_half_var > 0.5:
                    breaks['output_gaps'].append({
                        'region': region,
                        'potential_break_date': str(region_data.index[mid_point]),
                        'variance_change': float(second_half_var / first_half_var)
                    })
        
        return breaks
    
    def _get_default_data_sources(self) -> Dict[str, Any]:
        """Get default data source information."""
        return {
            'primary_source': 'Federal Reserve Economic Data (FRED)',
            'api_endpoint': 'https://api.stlouisfed.org/fred/',
            'data_vintage': 'Real-time and revised data',
            'regional_coverage': 'US states and metropolitan areas',
            'update_frequency': 'Monthly/Quarterly depending on series'
        }
    
    def _get_default_collection_info(self) -> Dict[str, Any]:
        """Get default data collection information."""
        return {
            'collection_method': 'Automated API retrieval',
            'data_processing': 'Minimal processing, seasonal adjustment as provided by source',
            'quality_checks': 'Automated outlier detection and missing value handling',
            'caching_strategy': 'Local caching with vintage tracking'
        }
    
    def _extract_vintage_info(self, regional_data: RegionalDataset) -> Dict[str, Any]:
        """Extract vintage information from regional dataset."""
        vintage_info = {
            'has_vintage_data': hasattr(regional_data, 'real_time_estimates'),
            'vintage_tracking': 'Enabled' if hasattr(regional_data, 'real_time_estimates') else 'Disabled'
        }
        
        if hasattr(regional_data, 'real_time_estimates') and regional_data.real_time_estimates:
            vintage_info['n_vintages'] = len(regional_data.real_time_estimates)
            vintage_info['vintage_dates'] = list(regional_data.real_time_estimates.keys())
        
        return vintage_info
    
    def _format_confidence_intervals(
        self, 
        regional_params: RegionalParameters, 
        param_name: str
    ) -> List[Dict[str, float]]:
        """Format confidence intervals for a parameter."""
        if param_name not in regional_params.confidence_intervals:
            return []
        
        lower, upper = regional_params.confidence_intervals[param_name]
        return [
            {'lower': float(lower[i]), 'upper': float(upper[i])}
            for i in range(len(lower))
        ]
    
    def _generate_diagnostic_metadata(self, regional_params: RegionalParameters) -> Dict[str, Any]:
        """Generate diagnostic test metadata."""
        return {
            'identification_tests': {
                'weak_identification': 'Tests for weak identification of parameters',
                'overidentification': 'Hansen J-test for overidentifying restrictions'
            },
            'specification_tests': {
                'spatial_autocorrelation': 'Moran\'s I test for residual spatial correlation',
                'parameter_stability': 'Tests for parameter stability across subsamples'
            },
            'robustness_checks': {
                'alternative_instruments': 'Results with alternative instrument sets',
                'subsample_stability': 'Parameter estimates for different time periods'
            }
        }
    
    def _generate_robustness_metadata(self) -> Dict[str, Any]:
        """Generate robustness check metadata."""
        return {
            'spatial_weights': {
                'alternative_constructions': 'Trade-only, distance-only, migration-only weights',
                'sensitivity_analysis': 'Parameter sensitivity to weight matrix specification'
            },
            'sample_periods': {
                'subsample_analysis': 'Estimation for different time periods',
                'crisis_periods': 'Separate analysis for financial crisis periods'
            },
            'model_specifications': {
                'alternative_lags': 'Different lag structures in spatial models',
                'parameter_restrictions': 'Results with and without parameter restrictions'
            }
        }
    
    def _compute_relative_contributions(self, policy_analysis: PolicyMistakeComponents) -> Dict[str, float]:
        """Compute relative contributions of mistake components."""
        total_abs = abs(policy_analysis.total_mistake)
        
        if total_abs == 0:
            return {
                'information_effect': 0.0,
                'weight_misallocation_effect': 0.0,
                'parameter_misspecification_effect': 0.0,
                'inflation_response_effect': 0.0
            }
        
        return {
            'information_effect': float(abs(policy_analysis.information_effect) / total_abs),
            'weight_misallocation_effect': float(abs(policy_analysis.weight_misallocation_effect) / total_abs),
            'parameter_misspecification_effect': float(abs(policy_analysis.parameter_misspecification_effect) / total_abs),
            'inflation_response_effect': float(abs(policy_analysis.inflation_response_effect) / total_abs)
        }
    
    def _generate_economic_interpretation(self, policy_analysis: PolicyMistakeComponents) -> Dict[str, str]:
        """Generate economic interpretation of policy analysis results."""
        interpretations = {}
        
        # Find dominant component
        components = {
            'information': abs(policy_analysis.information_effect),
            'weight_misallocation': abs(policy_analysis.weight_misallocation_effect),
            'parameter_misspec': abs(policy_analysis.parameter_misspecification_effect),
            'inflation_response': abs(policy_analysis.inflation_response_effect)
        }
        
        dominant = max(components, key=components.get)
        
        if dominant == 'information':
            interpretations['dominant_factor'] = (
                "Information limitations are the primary source of policy mistakes, "
                "suggesting that improved real-time data collection and processing "
                "could significantly enhance monetary policy effectiveness."
            )
        elif dominant == 'weight_misallocation':
            interpretations['dominant_factor'] = (
                "Regional weight misallocation is the main source of policy mistakes, "
                "indicating that the Federal Reserve's implicit regional weights "
                "differ significantly from welfare-optimal weights."
            )
        elif dominant == 'parameter_misspec':
            interpretations['dominant_factor'] = (
                "Parameter misspecification is the primary issue, suggesting that "
                "the Fed's model of the economy may not adequately capture "
                "regional heterogeneity in structural parameters."
            )
        else:
            interpretations['dominant_factor'] = (
                "Suboptimal inflation response is the main concern, indicating "
                "that the Fed's reaction to inflation may be too strong or too weak "
                "given regional economic conditions."
            )
        
        return interpretations
    
    def _generate_policy_implications(self, policy_analysis: PolicyMistakeComponents) -> List[str]:
        """Generate policy implications from analysis results."""
        implications = []
        
        # Check magnitude of total mistake
        if abs(policy_analysis.total_mistake) > 0.5:  # More than 50 basis points
            implications.append(
                "Large policy mistakes suggest significant room for improvement "
                "in monetary policy decision-making processes."
            )
        
        # Check information effect
        if abs(policy_analysis.information_effect) > 0.2:
            implications.append(
                "Substantial information effects indicate benefits from investing "
                "in improved real-time data collection and nowcasting capabilities."
            )
        
        # Check weight misallocation
        if abs(policy_analysis.weight_misallocation_effect) > 0.2:
            implications.append(
                "Significant weight misallocation suggests the need for more "
                "explicit consideration of regional heterogeneity in policy decisions."
            )
        
        return implications
    
    def _verify_welfare_ranking(self, scenarios: List[PolicyScenario]) -> Dict[str, Any]:
        """Verify the theoretical welfare ranking."""
        welfare_outcomes = {scenario.name: scenario.welfare_outcome for scenario in scenarios}
        
        # Expected ranking: Perfect Regional >= Perfect Info >= Optimal Regional >= Baseline
        expected_order = ['perfect_regional', 'perfect_info', 'optimal_regional', 'baseline']
        
        verification = {
            'expected_ranking_holds': True,
            'violations': []
        }
        
        # Check if we have the expected scenarios
        scenario_types = {scenario.scenario_type: scenario.welfare_outcome for scenario in scenarios}
        
        for i in range(len(expected_order) - 1):
            current_type = expected_order[i]
            next_type = expected_order[i + 1]
            
            if current_type in scenario_types and next_type in scenario_types:
                if scenario_types[current_type] < scenario_types[next_type]:
                    verification['expected_ranking_holds'] = False
                    verification['violations'].append(
                        f"{current_type} welfare ({scenario_types[current_type]:.6f}) "
                        f"< {next_type} welfare ({scenario_types[next_type]:.6f})"
                    )
        
        return verification
    
    def _generate_counterfactual_interpretation(self, scenarios: List[PolicyScenario]) -> Dict[str, str]:
        """Generate economic interpretation of counterfactual results."""
        welfare_outcomes = {scenario.name: scenario.welfare_outcome for scenario in scenarios}
        
        # Find best and worst scenarios
        best_scenario = max(welfare_outcomes, key=welfare_outcomes.get)
        worst_scenario = min(welfare_outcomes, key=welfare_outcomes.get)
        
        welfare_range = welfare_outcomes[best_scenario] - welfare_outcomes[worst_scenario]
        
        interpretation = {
            'welfare_range': (
                f"The welfare difference between the best ({best_scenario}) and "
                f"worst ({worst_scenario}) scenarios is {welfare_range:.6f}, "
                f"indicating {'substantial' if welfare_range > 0.001 else 'modest'} "
                f"potential gains from improved monetary policy."
            )
        }
        
        # Check for specific scenario insights
        if 'perfect_regional' in [s.name.lower().replace(' ', '_') for s in scenarios]:
            interpretation['perfect_regional'] = (
                "The Perfect Regional scenario represents the theoretical upper bound "
                "for welfare gains, showing the maximum benefits achievable with "
                "perfect information and optimal regional policy coordination."
            )
        
        return interpretation
    
    def _generate_policy_recommendations(self, scenarios: List[PolicyScenario]) -> List[str]:
        """Generate policy recommendations from counterfactual analysis."""
        recommendations = []
        
        welfare_outcomes = {scenario.name: scenario.welfare_outcome for scenario in scenarios}
        
        # Find baseline welfare
        baseline_welfare = next(
            (welfare for name, welfare in welfare_outcomes.items() if 'baseline' in name.lower()),
            min(welfare_outcomes.values())
        )
        
        # Check for significant improvements
        max_welfare = max(welfare_outcomes.values())
        potential_gain = max_welfare - baseline_welfare
        
        if potential_gain > 0.001:
            recommendations.append(
                f"Substantial welfare gains of {potential_gain:.6f} are achievable "
                "through improved monetary policy approaches, justifying investment "
                "in enhanced policy frameworks."
            )
        
        # Scenario-specific recommendations
        scenario_names = [s.name.lower().replace(' ', '_') for s in scenarios]
        
        if 'perfect_info' in scenario_names:
            perfect_info_welfare = next(
                welfare for name, welfare in welfare_outcomes.items() 
                if 'perfect' in name.lower() and 'info' in name.lower()
            )
            info_gain = perfect_info_welfare - baseline_welfare
            
            if info_gain > 0.0005:
                recommendations.append(
                    "Significant welfare gains from perfect information suggest "
                    "high returns to investing in improved real-time data systems "
                    "and nowcasting capabilities."
                )
        
        return recommendations
    
    def _get_scenario_description(self, scenario_type: str) -> str:
        """Get description for scenario type."""
        descriptions = {
            'baseline': 'Historical Federal Reserve policy decisions',
            'perfect_info': 'Optimal policy with perfect information about regional conditions',
            'optimal_regional': 'Optimal policy with Fed information set but welfare-optimal regional weights',
            'perfect_regional': 'Optimal policy with perfect information and optimal regional weights'
        }
        
        return descriptions.get(scenario_type, 'Unknown scenario type')
    
    def _compute_regional_impacts(self, scenario: PolicyScenario) -> Dict[str, Any]:
        """Compute regional impact statistics for a scenario."""
        if not hasattr(scenario, 'regional_outcomes') or scenario.regional_outcomes is None:
            return {}
        
        regional_impacts = {}
        
        # Compute welfare losses by region if available
        if 'welfare_loss' in scenario.regional_outcomes.columns:
            welfare_losses = scenario.regional_outcomes['welfare_loss']
            regional_impacts['welfare_statistics'] = {
                'mean_welfare_loss': float(welfare_losses.mean()),
                'std_welfare_loss': float(welfare_losses.std()),
                'max_welfare_loss': float(welfare_losses.max()),
                'min_welfare_loss': float(welfare_losses.min())
            }
        
        return regional_impacts
    
    def _generate_reproducibility_info(self, analysis_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reproducibility information."""
        return {
            'software_environment': {
                'python_version': '3.8+',
                'key_dependencies': [
                    'pandas>=1.3.0',
                    'numpy>=1.21.0',
                    'plotly>=5.0.0',
                    'scipy>=1.7.0'
                ],
                'random_seed': analysis_config.get('random_seed') if analysis_config else None
            },
            'computational_environment': {
                'estimation_tolerance': analysis_config.get('estimation', {}).get('tolerance', 1e-6) if analysis_config else 1e-6,
                'max_iterations': analysis_config.get('estimation', {}).get('max_iterations', 1000) if analysis_config else 1000,
                'parallel_processing': analysis_config.get('parallel', False) if analysis_config else False
            },
            'data_requirements': {
                'minimum_time_periods': 20,
                'minimum_regions': 3,
                'required_variables': ['output_gap', 'inflation_rate', 'interest_rate']
            }
        }


class DocumentationGenerator:
    """
    Generates comprehensive documentation for analysis workflows.
    
    Creates user guides, API documentation, and methodological
    documentation for the regional monetary policy analysis framework.
    """
    
    def __init__(self, output_dir: str = "documentation"):
        """
        Initialize documentation generator.
        
        Args:
            output_dir: Directory for documentation files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_user_guide(self) -> str:
        """
        Generate comprehensive user guide.
        
        Returns:
            Path to generated user guide
        """
        user_guide_content = """
# Regional Monetary Policy Analysis Framework - User Guide

## Overview

This framework provides comprehensive tools for analyzing regional heterogeneity in monetary policy transmission, estimating structural parameters across different regions, and evaluating policy effectiveness using real-time data.

## Getting Started

### Installation and Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Obtain FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html

3. Set up configuration:
   ```python
   from regional_monetary_policy.presentation.api import RegionalMonetaryPolicyAPI
   
   api = RegionalMonetaryPolicyAPI(fred_api_key="your_api_key_here")
   ```

### Basic Usage

#### 1. Load Regional Data

```python
# Define regions and time period
regions = ['CA', 'TX', 'NY', 'FL']  # State codes
start_date = '2000-01-01'
end_date = '2020-12-31'

# Load data
regional_data = api.load_regional_data(regions, start_date, end_date)
```

#### 2. Estimate Parameters

```python
# Estimate regional structural parameters
regional_params = api.estimate_parameters()
```

#### 3. Analyze Policy Mistakes

```python
# Load Fed policy rates (example)
fed_rates = pd.Series(...)  # Your Fed policy rate data

# Analyze policy mistakes
policy_analysis = api.analyze_policy_mistakes(fed_rates)
```

#### 4. Run Counterfactual Analysis

```python
# Run counterfactual scenarios
scenarios = api.run_counterfactual_analysis()
```

#### 5. Generate Results

```python
# Create visualizations
figures = api.create_visualizations()

# Export results
exported_files = api.export_results()

# Generate reports
reports = api.generate_reports()
```

## Advanced Usage

### Custom Analysis Workflows

```python
from regional_monetary_policy.presentation.api import AnalysisWorkflow

# Create workflow instance
workflow = AnalysisWorkflow(api)

# Run complete analysis
results = workflow.full_analysis_workflow(
    regions=['CA', 'TX', 'NY'],
    start_date='2000-01-01',
    end_date='2020-12-31',
    fed_policy_rates=fed_rates
)
```

### Parameter Sensitivity Analysis

```python
# Define parameter variations
variations = {
    'gmm_tolerance': [1e-4, 1e-5, 1e-6],
    'spatial_weight_threshold': [0.1, 0.05, 0.01]
}

# Run sensitivity analysis
sensitivity_results = workflow.parameter_sensitivity_analysis(variations)
```

### Robustness Checks

```python
# Define alternative specifications
alt_specs = [
    {
        'name': 'Trade-only weights',
        'spatial_config': {'use_trade_only': True}
    },
    {
        'name': 'Distance-only weights',
        'spatial_config': {'use_distance_only': True}
    }
]

# Run robustness checks
robustness_results = workflow.robustness_check_workflow(alt_specs)
```

## Output Formats

### Data Export Formats

- **CSV**: Tabular data suitable for Excel or statistical software
- **JSON**: Structured data for programmatic access
- **LaTeX**: Publication-ready tables for academic papers

### Chart Export Formats

- **PNG**: High-resolution raster images
- **PDF**: Vector graphics for publications
- **SVG**: Scalable vector graphics for web use
- **HTML**: Interactive charts for presentations

### Report Formats

- **HTML**: Interactive reports with embedded charts
- **PDF**: Static reports for distribution
- **LaTeX**: Source files for academic papers

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your FRED API key is valid and has sufficient quota
2. **Data Availability**: Some regional series may have limited historical data
3. **Estimation Convergence**: Try adjusting tolerance or maximum iterations
4. **Memory Issues**: Use data subsets for initial testing with large datasets

### Performance Optimization

- Use caching for repeated data requests
- Enable parallel processing for parameter estimation
- Limit time periods for initial exploratory analysis
- Use data subsets for testing and development

## Best Practices

### Data Quality

- Always check data completeness before estimation
- Handle missing values appropriately
- Validate outliers and structural breaks
- Use real-time data vintages when available

### Model Specification

- Test multiple spatial weight constructions
- Perform robustness checks with alternative specifications
- Validate identification assumptions
- Check parameter stability across subsamples

### Result Interpretation

- Consider economic significance alongside statistical significance
- Validate welfare rankings in counterfactual analysis
- Interpret policy mistake decomposition in economic terms
- Consider uncertainty in parameter estimates

## References

- Theoretical framework: [Mathematical Appendix]
- FRED API documentation: https://fred.stlouisfed.org/docs/api/
- Spatial econometrics: Anselin (1988), LeSage & Pace (2009)
- Regional monetary policy: Carlino & DeFina (1998), Owyang & Wall (2009)
"""
        
        user_guide_file = self.output_dir / "user_guide.md"
        with open(user_guide_file, 'w') as f:
            f.write(user_guide_content)
        
        return str(user_guide_file)
    
    def generate_api_documentation(self) -> str:
        """
        Generate API documentation.
        
        Returns:
            Path to generated API documentation
        """
        api_doc_content = """
# Regional Monetary Policy Analysis API Documentation

## RegionalMonetaryPolicyAPI Class

The main API class providing programmatic access to all analysis components.

### Constructor

```python
RegionalMonetaryPolicyAPI(fred_api_key, cache_dir="data/cache", output_dir="output")
```

**Parameters:**
- `fred_api_key` (str): FRED API key for data access
- `cache_dir` (str): Directory for data caching
- `output_dir` (str): Directory for output files

### Methods

#### load_regional_data()

```python
load_regional_data(regions, start_date, end_date, indicators=None)
```

Load regional economic data from FRED.

**Parameters:**
- `regions` (List[str]): List of region identifiers
- `start_date` (str): Start date (YYYY-MM-DD format)
- `end_date` (str): End date (YYYY-MM-DD format)
- `indicators` (List[str], optional): List of economic indicators

**Returns:**
- `RegionalDataset`: Regional dataset with loaded data

#### estimate_parameters()

```python
estimate_parameters(spatial_weights_config=None, estimation_config=None)
```

Estimate regional structural parameters.

**Parameters:**
- `spatial_weights_config` (Dict, optional): Spatial weight construction config
- `estimation_config` (Dict, optional): Parameter estimation config

**Returns:**
- `RegionalParameters`: Estimated regional parameters

#### analyze_policy_mistakes()

```python
analyze_policy_mistakes(fed_policy_rates, real_time_data=None)
```

Analyze policy mistakes and decompose into components.

**Parameters:**
- `fed_policy_rates` (pd.Series): Historical Fed policy rates
- `real_time_data` (RegionalDataset, optional): Real-time data available to Fed

**Returns:**
- `PolicyMistakeComponents`: Policy mistake decomposition results

#### run_counterfactual_analysis()

```python
run_counterfactual_analysis(scenarios=None)
```

Run counterfactual policy analysis.

**Parameters:**
- `scenarios` (List[str], optional): List of scenario names to analyze

**Returns:**
- `List[PolicyScenario]`: List of policy scenarios with results

#### create_visualizations()

```python
create_visualizations(visualization_types=None)
```

Create standard visualizations for analysis results.

**Parameters:**
- `visualization_types` (List[str], optional): List of visualization types

**Returns:**
- `Dict[str, Any]`: Dictionary mapping visualization names to Plotly figures

#### export_results()

```python
export_results(export_formats=None, include_charts=True)
```

Export all analysis results in specified formats.

**Parameters:**
- `export_formats` (List[str], optional): List of export formats
- `include_charts` (bool): Whether to export charts

**Returns:**
- `Dict[str, Any]`: Dictionary with paths to exported files

#### generate_reports()

```python
generate_reports(report_types=None)
```

Generate comprehensive analysis reports.

**Parameters:**
- `report_types` (List[str], optional): List of report types to generate

**Returns:**
- `Dict[str, str]`: Dictionary mapping report types to file paths

## Data Models

### RegionalDataset

Container for regional economic data.

**Attributes:**
- `output_gaps` (pd.DataFrame): Regional output gaps
- `inflation_rates` (pd.DataFrame): Regional inflation rates
- `interest_rates` (pd.Series): Policy interest rates
- `metadata` (Dict): Dataset metadata

### RegionalParameters

Container for estimated regional structural parameters.

**Attributes:**
- `sigma` (np.ndarray): Interest rate sensitivities
- `kappa` (np.ndarray): Phillips curve slopes
- `psi` (np.ndarray): Demand spillover parameters
- `phi` (np.ndarray): Price spillover parameters
- `beta` (np.ndarray): Discount factors
- `standard_errors` (Dict): Parameter standard errors
- `confidence_intervals` (Dict): Parameter confidence intervals

### PolicyMistakeComponents

Container for policy mistake decomposition results.

**Attributes:**
- `total_mistake` (float): Total policy mistake
- `information_effect` (float): Information effect component
- `weight_misallocation_effect` (float): Weight misallocation component
- `parameter_misspecification_effect` (float): Parameter misspecification component
- `inflation_response_effect` (float): Inflation response component

### PolicyScenario

Container for counterfactual policy scenario.

**Attributes:**
- `name` (str): Scenario name
- `policy_rates` (pd.Series): Policy rates under scenario
- `regional_outcomes` (pd.DataFrame): Regional economic outcomes
- `welfare_outcome` (float): Welfare outcome under scenario
- `scenario_type` (str): Type of scenario

## Export and Reporting Classes

### DataExporter

Handles multi-format data export functionality.

### ChartExporter

Handles high-resolution chart export for publication-quality figures.

### ReportGenerator

Generate comprehensive reports for regional monetary policy analysis.

### MetadataGenerator

Generates comprehensive metadata for analysis results.

## Workflow Classes

### AnalysisWorkflow

Pre-defined analysis workflows for common research questions.

#### full_analysis_workflow()

Run complete analysis workflow from data loading to reporting.

#### parameter_sensitivity_analysis()

Run parameter sensitivity analysis.

#### robustness_check_workflow()

Run robustness checks with alternative model specifications.

## Error Handling

The API includes comprehensive error handling for:
- Invalid API keys or network issues
- Data availability problems
- Estimation convergence failures
- Invalid parameter specifications

All methods include appropriate error messages and suggested solutions.

## Performance Considerations

- Use caching for repeated data requests
- Enable parallel processing where available
- Monitor memory usage with large datasets
- Respect FRED API rate limits (120 calls/minute)
"""
        
        api_doc_file = self.output_dir / "api_documentation.md"
        with open(api_doc_file, 'w') as f:
            f.write(api_doc_content)
        
        return str(api_doc_file)
    
    def generate_methodology_documentation(self) -> str:
        """
        Generate detailed methodology documentation.
        
        Returns:
            Path to generated methodology documentation
        """
        methodology_content = """
# Regional Monetary Policy Analysis - Methodology Documentation

## Theoretical Framework

### Multi-Region New Keynesian DSGE Model

The analysis is based on a multi-region New Keynesian Dynamic Stochastic General Equilibrium (DSGE) model with spatial spillovers. The model incorporates:

1. **Regional Heterogeneity**: Different structural parameters across regions
2. **Spatial Spillovers**: Economic linkages through trade, migration, and financial flows
3. **Monetary Policy Transmission**: Regional variation in policy effectiveness
4. **Welfare Analysis**: Social welfare function with regional weights

### Key Equations

#### Regional IS Curve
```
x_{i,t} = E_t[x_{i,t+1}] - σ_i^{-1}(r_{t} - E_t[π_{i,t+1}]) + ψ_i ∑_j W_{ij} x_{j,t} + u_{i,t}
```

Where:
- x_{i,t}: Output gap in region i at time t
- σ_i: Interest rate sensitivity (inverse of intertemporal elasticity)
- r_t: Real interest rate
- π_{i,t}: Inflation in region i
- ψ_i: Demand spillover parameter
- W_{ij}: Spatial weight between regions i and j

#### Regional Phillips Curve
```
π_{i,t} = β E_t[π_{i,t+1}] + κ_i x_{i,t} + φ_i ∑_j W_{ij} π_{j,t} + v_{i,t}
```

Where:
- κ_i: Phillips curve slope (related to price stickiness)
- φ_i: Price spillover parameter
- β: Discount factor

#### Monetary Policy Rule
```
r_t = ρ_r r_{t-1} + (1-ρ_r)[φ_π π_t^{agg} + φ_x x_t^{agg}] + ε_t
```

Where aggregate variables are weighted averages of regional variables.

## Estimation Methodology

### Three-Stage Estimation Procedure

#### Stage 1: Spatial Weight Matrix Construction

Spatial weights W_{ij} are constructed as a weighted combination of:

1. **Trade Flows**: Bilateral trade between regions
2. **Migration Patterns**: Population flows between regions  
3. **Financial Linkages**: Banking and financial connections
4. **Geographic Distance**: Inverse distance weighting

The spatial weight matrix is normalized to have row sums equal to one:
```
W_{ij} = α₁ Trade_{ij} + α₂ Migration_{ij} + α₃ Financial_{ij} + α₄ Distance_{ij}^{-1}
```

#### Stage 2: Regional Parameter Estimation

Regional structural parameters (σᵢ, κᵢ, ψᵢ, φᵢ) are estimated using Generalized Method of Moments (GMM) with the following moment conditions:

1. **IS Curve Moments**: Orthogonality of demand shocks with lagged variables
2. **Phillips Curve Moments**: Orthogonality of supply shocks with lagged variables
3. **Spatial Moments**: Consistency of spatial spillover parameters

#### Stage 3: Policy Parameter Estimation

Federal Reserve reaction function parameters are estimated using:
1. Historical policy decisions
2. Real-time data available to policymakers
3. Implicit regional weights in policy decisions

### Identification Strategy

Parameter identification relies on:

1. **Regional Variation**: Cross-sectional differences in economic conditions
2. **Temporal Variation**: Time-series variation in regional and aggregate variables
3. **Spatial Exclusion Restrictions**: Assumption that direct spillovers occur only between connected regions
4. **Policy Exogeneity**: Monetary policy responds to aggregate, not region-specific, conditions

### Diagnostic Tests

#### Weak Identification Tests
- Kleibergen-Paap F-statistic for weak instruments
- Stock-Yogo critical values for inference robustness

#### Overidentification Tests  
- Hansen J-test for overidentifying restrictions
- Difference-in-Hansen tests for subset orthogonality

#### Spatial Specification Tests
- Moran's I test for residual spatial autocorrelation
- LM tests for spatial lag vs. spatial error specifications

## Policy Analysis Framework

### Optimal Policy Derivation

The welfare-maximizing monetary policy rate is derived by minimizing the social loss function:

```
L_t = ∑_i ω_i [λ_i x_{i,t}² + π_{i,t}²]
```

Where:
- ω_i: Regional welfare weights
- λ_i: Regional preference parameters

This yields the optimal policy rule:
```
r_t^* = φ_π^* π_t^{agg} + φ_x^* x_t^{agg}
```

With optimal coefficients derived from regional parameters and welfare weights.

### Policy Mistake Decomposition (Theorem 4)

The difference between actual and optimal policy is decomposed as:

```
r_t - r_t^* = Information Effect + Weight Misallocation + Parameter Misspecification + Inflation Response
```

#### Information Effect
Captures the impact of imperfect information about regional conditions:
```
Info_t = (φ_π^* - φ_π^{Fed}) (π_t^{RT} - π_t^{True})
```

#### Weight Misallocation Effect  
Captures suboptimal regional weights in policy decisions:
```
Weight_t = ∑_i (ω_i^{Fed} - ω_i^*) [λ_i x_{i,t} + π_{i,t}]
```

#### Parameter Misspecification Effect
Captures incorrect structural parameter assumptions:
```
Param_t = f(θ^{Fed} - θ^{True})
```

#### Inflation Response Effect
Captures suboptimal inflation response coefficient:
```
Inflation_t = (φ_π^{Fed} - φ_π^*) π_t^{agg}
```

## Counterfactual Analysis

### Policy Scenarios

#### Baseline (B)
Historical Federal Reserve policy decisions with actual information sets.

#### Perfect Information (PI)  
Optimal policy with perfect information about regional conditions but Fed's regional weights and parameters.

#### Optimal Regional (OR)
Optimal policy with Fed's information set but welfare-optimal regional weights and parameters.

#### Perfect Regional (PR)
Optimal policy with perfect information and welfare-optimal regional weights and parameters.

### Welfare Evaluation

Social welfare under each scenario is computed as:
```
W^s = -E[∑_t ∑_i ω_i^* [λ_i x_{i,t}^s² + π_{i,t}^s²]]
```

The theoretical welfare ranking is: W^{PR} ≥ W^{PI} ≥ W^{OR} ≥ W^B

### Welfare Decomposition

Welfare gains are decomposed into:
1. **Information Gains**: W^{PI} - W^B and W^{PR} - W^{OR}
2. **Regional Optimization Gains**: W^{OR} - W^B and W^{PR} - W^{PI}
3. **Total Potential Gains**: W^{PR} - W^B

## Data Requirements and Sources

### Regional Economic Data
- **Output Gaps**: State-level estimates from FRED
- **Inflation Rates**: Regional CPI or PCE deflators
- **Employment**: Regional unemployment rates
- **Income**: Regional personal income growth

### Spatial Linkage Data
- **Trade Flows**: Commodity Flow Survey (Census Bureau)
- **Migration**: American Community Survey (Census Bureau)  
- **Financial Links**: Summary of Deposits (FDIC)
- **Distance**: Geographic centroids and great circle distances

### Policy Data
- **Federal Funds Rate**: FRED series FEDFUNDS
- **FOMC Minutes**: Real-time information sets
- **Economic Projections**: Summary of Economic Projections

## Robustness Checks

### Alternative Specifications
1. **Spatial Weights**: Trade-only, distance-only, migration-only
2. **Sample Periods**: Pre-crisis, post-crisis, full sample
3. **Regional Groupings**: States vs. metropolitan areas
4. **Lag Structures**: Different lag lengths in spatial models

### Sensitivity Analysis
1. **Parameter Restrictions**: Homogeneous vs. heterogeneous parameters
2. **Instrument Sets**: Alternative instrument choices
3. **Estimation Methods**: 2SLS vs. GMM vs. Maximum Likelihood
4. **Spatial Specifications**: Spatial lag vs. spatial error vs. spatial Durbin

## Computational Implementation

### Optimization Algorithms
- **GMM Estimation**: Iterative GMM with optimal weighting matrix
- **Spatial Weights**: Constrained optimization for weight normalization
- **Policy Optimization**: Numerical optimization of welfare function

### Numerical Considerations
- **Convergence Criteria**: Tolerance levels for parameter estimates
- **Starting Values**: Multiple starting points for global optimization
- **Standard Errors**: Bootstrap or asymptotic standard errors
- **Parallel Processing**: Multi-core estimation for robustness checks

### Software Requirements
- **Python 3.8+**: Core programming language
- **NumPy/SciPy**: Numerical computation and optimization
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization
- **Statsmodels**: Statistical estimation and testing

## References

### Theoretical Literature
- Clarida, R., Galí, J., & Gertler, M. (1999). The science of monetary policy
- Woodford, M. (2003). Interest and prices: Foundations of a theory of monetary policy
- Galí, J. (2015). Monetary policy, inflation, and the business cycle

### Regional Monetary Policy
- Carlino, G., & DeFina, R. (1998). The differential regional effects of monetary policy
- Owyang, M. T., & Wall, H. J. (2009). Regional VARs and the channels of monetary policy
- Fratantoni, M., & Schuh, S. (2003). Monetary policy, housing, and heterogeneous regional markets

### Spatial Econometrics  
- Anselin, L. (1988). Spatial econometrics: Methods and models
- LeSage, J., & Pace, R. K. (2009). Introduction to spatial econometrics
- Elhorst, J. P. (2014). Spatial econometrics: From cross-sectional data to spatial panels

### Estimation Methods
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression
"""
        
        methodology_file = self.output_dir / "methodology_documentation.md"
        with open(methodology_file, 'w') as f:
            f.write(methodology_content)
        
        return str(methodology_file)