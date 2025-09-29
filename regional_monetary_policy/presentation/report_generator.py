"""
Report generation and export functionality for regional monetary policy analysis.

This module provides comprehensive report generation, data export, and documentation
capabilities for sharing analysis results and integrating findings into academic
papers or policy documents.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
import plotly.graph_objects as go
import plotly.io as pio
from dataclasses import asdict

from ..data.models import RegionalDataset, ValidationReport
from ..econometric.models import RegionalParameters, EstimationResults
from ..policy.models import (
    PolicyScenario, PolicyMistakeComponents, 
    ComparisonResults, WelfareDecomposition
)
from ..config.settings import AnalysisSettings


class DataExporter:
    """
    Handles multi-format data export functionality.
    
    Supports export to CSV, JSON, LaTeX tables, and other formats
    suitable for academic and policy research.
    """
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_regional_data(
        self, 
        regional_data: RegionalDataset,
        filename_prefix: str = "regional_data",
        formats: List[str] = ["csv", "json"]
    ) -> Dict[str, str]:
        """
        Export regional economic data in multiple formats.
        
        Args:
            regional_data: Regional dataset to export
            filename_prefix: Prefix for output filenames
            formats: List of export formats ("csv", "json", "latex")
            
        Returns:
            Dictionary mapping format to output filepath
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in formats:
            if fmt == "csv":
                # Export output gaps
                output_gaps_file = self.output_dir / f"{filename_prefix}_output_gaps_{timestamp}.csv"
                regional_data.output_gaps.to_csv(output_gaps_file)
                
                # Export inflation rates
                inflation_file = self.output_dir / f"{filename_prefix}_inflation_{timestamp}.csv"
                regional_data.inflation_rates.to_csv(inflation_file)
                
                # Export interest rates
                interest_file = self.output_dir / f"{filename_prefix}_interest_rates_{timestamp}.csv"
                regional_data.interest_rates.to_csv(interest_file)
                
                exported_files["csv"] = {
                    "output_gaps": str(output_gaps_file),
                    "inflation_rates": str(inflation_file),
                    "interest_rates": str(interest_file)
                }
                
            elif fmt == "json":
                json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
                
                # Convert to JSON-serializable format
                data_dict = {
                    "output_gaps": {
                        str(region): {str(date): value for date, value in series.items()}
                        for region, series in regional_data.output_gaps.iterrows()
                    },
                    "inflation_rates": {
                        str(region): {str(date): value for date, value in series.items()}
                        for region, series in regional_data.inflation_rates.iterrows()
                    },
                    "interest_rates": {str(date): value for date, value in regional_data.interest_rates.items()},
                    "metadata": regional_data.metadata,
                    "export_timestamp": timestamp
                }
                
                with open(json_file, 'w') as f:
                    json.dump(data_dict, f, indent=2, default=str)
                
                exported_files["json"] = str(json_file)
                
            elif fmt == "latex":
                latex_file = self.output_dir / f"{filename_prefix}_{timestamp}.tex"
                
                # Create LaTeX table for summary statistics
                summary_stats = self._compute_summary_statistics(regional_data)
                latex_content = self._create_latex_table(
                    summary_stats, 
                    "Regional Economic Data Summary",
                    "tab:regional_summary"
                )
                
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
                
                exported_files["latex"] = str(latex_file)
        
        return exported_files
    
    def export_parameter_estimates(
        self,
        regional_params: RegionalParameters,
        filename_prefix: str = "parameter_estimates",
        formats: List[str] = ["csv", "json", "latex"]
    ) -> Dict[str, str]:
        """
        Export parameter estimation results in multiple formats.
        
        Args:
            regional_params: Regional parameter estimates
            filename_prefix: Prefix for output filenames
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to output filepath
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get parameter summary
        param_summary = regional_params.get_parameter_summary()
        
        for fmt in formats:
            if fmt == "csv":
                csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
                param_summary.to_csv(csv_file)
                exported_files["csv"] = str(csv_file)
                
            elif fmt == "json":
                json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
                
                # Convert to JSON format
                param_dict = {
                    "sigma": regional_params.sigma.tolist(),
                    "kappa": regional_params.kappa.tolist(),
                    "psi": regional_params.psi.tolist(),
                    "phi": regional_params.phi.tolist(),
                    "beta": regional_params.beta.tolist(),
                    "standard_errors": {k: v.tolist() for k, v in regional_params.standard_errors.items()},
                    "confidence_intervals": {
                        k: {"lower": v[0].tolist(), "upper": v[1].tolist()} 
                        for k, v in regional_params.confidence_intervals.items()
                    },
                    "export_timestamp": timestamp
                }
                
                with open(json_file, 'w') as f:
                    json.dump(param_dict, f, indent=2)
                
                exported_files["json"] = str(json_file)
                
            elif fmt == "latex":
                latex_file = self.output_dir / f"{filename_prefix}_{timestamp}.tex"
                
                # Create LaTeX table for parameter estimates
                latex_content = self._create_parameter_latex_table(regional_params)
                
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
                
                exported_files["latex"] = str(latex_file)
        
        return exported_files
    
    def export_policy_analysis(
        self,
        mistake_components: PolicyMistakeComponents,
        filename_prefix: str = "policy_analysis",
        formats: List[str] = ["csv", "json", "latex"]
    ) -> Dict[str, str]:
        """
        Export policy mistake decomposition results.
        
        Args:
            mistake_components: Policy mistake decomposition
            filename_prefix: Prefix for output filenames
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to output filepath
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame for easier export
        components_df = pd.DataFrame({
            'Component': [
                'Total Mistake',
                'Information Effect', 
                'Weight Misallocation',
                'Parameter Misspecification',
                'Inflation Response'
            ],
            'Value': [
                mistake_components.total_mistake,
                mistake_components.information_effect,
                mistake_components.weight_misallocation_effect,
                mistake_components.parameter_misspecification_effect,
                mistake_components.inflation_response_effect
            ]
        })
        
        for fmt in formats:
            if fmt == "csv":
                csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
                components_df.to_csv(csv_file, index=False)
                exported_files["csv"] = str(csv_file)
                
            elif fmt == "json":
                json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
                
                # Convert to JSON format
                components_dict = asdict(mistake_components)
                components_dict["export_timestamp"] = timestamp
                
                with open(json_file, 'w') as f:
                    json.dump(components_dict, f, indent=2)
                
                exported_files["json"] = str(json_file)
                
            elif fmt == "latex":
                latex_file = self.output_dir / f"{filename_prefix}_{timestamp}.tex"
                
                # Create LaTeX table for policy analysis
                latex_content = self._create_latex_table(
                    components_df.set_index('Component'),
                    "Policy Mistake Decomposition",
                    "tab:policy_mistakes"
                )
                
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
                
                exported_files["latex"] = str(latex_file)
        
        return exported_files
    
    def export_counterfactual_results(
        self,
        scenarios: List[PolicyScenario],
        comparison_results: ComparisonResults,
        filename_prefix: str = "counterfactual_analysis",
        formats: List[str] = ["csv", "json", "latex"]
    ) -> Dict[str, str]:
        """
        Export counterfactual analysis results.
        
        Args:
            scenarios: List of policy scenarios
            comparison_results: Scenario comparison results
            filename_prefix: Prefix for output filenames
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to output filepath
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary DataFrame
        summary_df = comparison_results.summary_table()
        
        for fmt in formats:
            if fmt == "csv":
                # Export summary table
                summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.csv"
                summary_df.to_csv(summary_file, index=False)
                
                # Export detailed scenario data
                scenario_files = {}
                for scenario in scenarios:
                    scenario_file = self.output_dir / f"{filename_prefix}_{scenario.name.lower().replace(' ', '_')}_{timestamp}.csv"
                    
                    # Combine policy rates and regional outcomes
                    scenario_data = pd.DataFrame({
                        'policy_rate': scenario.policy_rates
                    })
                    
                    # Add regional outcomes if available
                    if hasattr(scenario, 'regional_outcomes') and scenario.regional_outcomes is not None:
                        for col in scenario.regional_outcomes.columns:
                            scenario_data[col] = scenario.regional_outcomes[col]
                    
                    scenario_data.to_csv(scenario_file)
                    scenario_files[scenario.name] = str(scenario_file)
                
                exported_files["csv"] = {
                    "summary": str(summary_file),
                    "scenarios": scenario_files
                }
                
            elif fmt == "json":
                json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
                
                # Convert scenarios to JSON format
                scenarios_dict = {}
                for scenario in scenarios:
                    scenarios_dict[scenario.name] = {
                        "policy_rates": scenario.policy_rates.to_dict(),
                        "welfare_outcome": scenario.welfare_outcome,
                        "scenario_type": scenario.scenario_type
                    }
                    
                    if hasattr(scenario, 'regional_outcomes') and scenario.regional_outcomes is not None:
                        scenarios_dict[scenario.name]["regional_outcomes"] = scenario.regional_outcomes.to_dict()
                
                export_data = {
                    "scenarios": scenarios_dict,
                    "summary": summary_df.to_dict('records'),
                    "export_timestamp": timestamp
                }
                
                with open(json_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                exported_files["json"] = str(json_file)
                
            elif fmt == "latex":
                latex_file = self.output_dir / f"{filename_prefix}_{timestamp}.tex"
                
                # Create LaTeX table for counterfactual results
                latex_content = self._create_latex_table(
                    summary_df.set_index('Scenario'),
                    "Counterfactual Policy Analysis Results",
                    "tab:counterfactual_results"
                )
                
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
                
                exported_files["latex"] = str(latex_file)
        
        return exported_files
    
    def _compute_summary_statistics(self, regional_data: RegionalDataset) -> pd.DataFrame:
        """Compute summary statistics for regional data."""
        summary_stats = []
        
        # Output gap statistics
        for region in regional_data.output_gaps.index:
            region_data = regional_data.output_gaps.loc[region]
            summary_stats.append({
                'Region': region,
                'Variable': 'Output Gap',
                'Mean': region_data.mean(),
                'Std': region_data.std(),
                'Min': region_data.min(),
                'Max': region_data.max()
            })
        
        # Inflation statistics
        for region in regional_data.inflation_rates.index:
            region_data = regional_data.inflation_rates.loc[region]
            summary_stats.append({
                'Region': region,
                'Variable': 'Inflation Rate',
                'Mean': region_data.mean(),
                'Std': region_data.std(),
                'Min': region_data.min(),
                'Max': region_data.max()
            })
        
        return pd.DataFrame(summary_stats)
    
    def _create_latex_table(
        self, 
        df: pd.DataFrame, 
        caption: str, 
        label: str,
        precision: int = 4
    ) -> str:
        """Create a LaTeX table from DataFrame."""
        
        # Format numbers to specified precision
        formatted_df = df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype in [np.float64, np.float32]:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.{precision}f}")
        
        # Generate LaTeX table
        latex_table = formatted_df.to_latex(
            escape=False,
            column_format='l' + 'c' * len(df.columns),
            caption=caption,
            label=label
        )
        
        return latex_table
    
    def _create_parameter_latex_table(self, regional_params: RegionalParameters) -> str:
        """Create specialized LaTeX table for parameter estimates."""
        
        # Create parameter table with confidence intervals
        param_data = []
        regions = [f"Region {i+1}" for i in range(regional_params.n_regions)]
        
        for i, region in enumerate(regions):
            row = {
                'Region': region,
                'σ (Interest Rate Sensitivity)': f"{regional_params.sigma[i]:.4f}",
                'κ (Phillips Curve Slope)': f"{regional_params.kappa[i]:.4f}",
                'ψ (Demand Spillover)': f"{regional_params.psi[i]:.4f}",
                'φ (Price Spillover)': f"{regional_params.phi[i]:.4f}",
                'β (Discount Factor)': f"{regional_params.beta[i]:.4f}"
            }
            
            # Add confidence intervals if available
            if 'sigma' in regional_params.confidence_intervals:
                lower, upper = regional_params.confidence_intervals['sigma']
                row['σ (Interest Rate Sensitivity)'] += f" [{lower[i]:.4f}, {upper[i]:.4f}]"
            
            param_data.append(row)
        
        param_df = pd.DataFrame(param_data)
        
        return self._create_latex_table(
            param_df.set_index('Region'),
            "Regional Parameter Estimates with 95\\% Confidence Intervals",
            "tab:parameter_estimates"
        )


class ChartExporter:
    """
    Handles high-resolution chart export for publication-quality figures.
    """
    
    def __init__(self, output_dir: str = "charts"):
        """
        Initialize chart exporter.
        
        Args:
            output_dir: Directory for exported charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set high-quality export settings
        self.export_settings = {
            'width': 1200,
            'height': 800,
            'scale': 2,  # For high DPI
            'format': 'png'
        }
    
    def export_figure(
        self,
        fig: go.Figure,
        filename: str,
        formats: List[str] = ["png", "pdf", "svg"],
        **kwargs
    ) -> Dict[str, str]:
        """
        Export Plotly figure in multiple high-resolution formats.
        
        Args:
            fig: Plotly figure to export
            filename: Base filename (without extension)
            formats: List of export formats
            **kwargs: Additional export settings
            
        Returns:
            Dictionary mapping format to output filepath
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Update export settings with any provided kwargs
        settings = {**self.export_settings, **kwargs}
        
        for fmt in formats:
            output_file = self.output_dir / f"{filename}_{timestamp}.{fmt}"
            
            try:
                if fmt == "png":
                    pio.write_image(
                        fig, 
                        output_file,
                        format="png",
                        width=settings['width'],
                        height=settings['height'],
                        scale=settings['scale']
                    )
                elif fmt == "pdf":
                    pio.write_image(
                        fig,
                        output_file,
                        format="pdf",
                        width=settings['width'],
                        height=settings['height']
                    )
                elif fmt == "svg":
                    pio.write_image(
                        fig,
                        output_file,
                        format="svg",
                        width=settings['width'],
                        height=settings['height']
                    )
                elif fmt == "html":
                    fig.write_html(str(output_file))
                
                exported_files[fmt] = str(output_file)
                
            except Exception as e:
                print(f"Warning: Failed to export {fmt} format: {e}")
        
        return exported_files
    
    def export_multiple_figures(
        self,
        figures: Dict[str, go.Figure],
        formats: List[str] = ["png", "pdf"]
    ) -> Dict[str, Dict[str, str]]:
        """
        Export multiple figures with consistent naming.
        
        Args:
            figures: Dictionary mapping figure names to Plotly figures
            formats: List of export formats
            
        Returns:
            Nested dictionary mapping figure names to format-filepath mappings
        """
        all_exports = {}
        
        for fig_name, fig in figures.items():
            exported_files = self.export_figure(fig, fig_name, formats)
            all_exports[fig_name] = exported_files
        
        return all_exports


class ReportGenerator:
    """
    Generate comprehensive reports for regional monetary policy analysis.
    
    Creates detailed analysis reports with methodology sections, results,
    and robustness checks suitable for academic and policy research.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exporters
        self.data_exporter = DataExporter(self.output_dir / "data")
        self.chart_exporter = ChartExporter(self.output_dir / "charts")
        
        # Setup Jinja2 environment for templates
        template_dir = Path(__file__).parent / "templates"
        if template_dir.exists():
            self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        else:
            self.jinja_env = None
    
    def generate_comprehensive_report(
        self,
        regional_data: RegionalDataset,
        regional_params: RegionalParameters,
        policy_analysis: PolicyMistakeComponents,
        counterfactual_results: List[PolicyScenario],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            regional_data: Regional economic dataset
            regional_params: Parameter estimation results
            policy_analysis: Policy mistake decomposition
            counterfactual_results: Counterfactual scenario results
            metadata: Additional metadata for the report
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"comprehensive_report_{timestamp}.html"
        
        # Prepare report data
        report_data = {
            'title': 'Regional Monetary Policy Analysis Report',
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'timestamp': timestamp,
            'metadata': metadata or {},
            'data_summary': self._create_data_summary(regional_data),
            'parameter_summary': self._create_parameter_summary(regional_params),
            'policy_analysis_summary': self._create_policy_summary(policy_analysis),
            'counterfactual_summary': self._create_counterfactual_summary(counterfactual_results)
        }
        
        # Generate HTML report
        if self.jinja_env:
            template = self.jinja_env.get_template('comprehensive_report.html')
            html_content = template.render(**report_data)
        else:
            html_content = self._create_basic_html_report(report_data)
        
        # Write report file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_file)
    
    def generate_methodology_report(
        self,
        estimation_config: Dict[str, Any],
        model_specification: Dict[str, Any]
    ) -> str:
        """
        Generate a detailed methodology report.
        
        Args:
            estimation_config: Configuration used for estimation
            model_specification: Model specification details
            
        Returns:
            Path to methodology report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        methodology_file = self.output_dir / f"methodology_report_{timestamp}.html"
        
        # Prepare methodology content
        methodology_data = {
            'title': 'Regional Monetary Policy Analysis Methodology',
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'estimation_config': estimation_config,
            'model_specification': model_specification,
            'theoretical_framework': self._get_theoretical_framework_text(),
            'estimation_procedure': self._get_estimation_procedure_text(),
            'identification_strategy': self._get_identification_strategy_text()
        }
        
        # Generate HTML methodology report
        if self.jinja_env:
            template = self.jinja_env.get_template('methodology_report.html')
            html_content = template.render(**methodology_data)
        else:
            html_content = self._create_basic_methodology_html(methodology_data)
        
        # Write methodology file
        with open(methodology_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(methodology_file)
    
    def generate_executive_summary(
        self,
        key_findings: Dict[str, Any],
        policy_implications: List[str],
        welfare_gains: Dict[str, float]
    ) -> str:
        """
        Generate an executive summary report.
        
        Args:
            key_findings: Dictionary of key research findings
            policy_implications: List of policy implications
            welfare_gains: Welfare gains from different scenarios
            
        Returns:
            Path to executive summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"executive_summary_{timestamp}.html"
        
        # Prepare summary content
        summary_data = {
            'title': 'Executive Summary: Regional Monetary Policy Analysis',
            'generation_date': datetime.now().strftime("%B %d, %Y"),
            'key_findings': key_findings,
            'policy_implications': policy_implications,
            'welfare_gains': welfare_gains,
            'main_conclusions': self._extract_main_conclusions(key_findings, welfare_gains)
        }
        
        # Generate HTML summary
        if self.jinja_env:
            template = self.jinja_env.get_template('executive_summary.html')
            html_content = template.render(**summary_data)
        else:
            html_content = self._create_basic_summary_html(summary_data)
        
        # Write summary file
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(summary_file)
    
    def _create_data_summary(self, regional_data: RegionalDataset) -> Dict[str, Any]:
        """Create summary of regional data for report."""
        return {
            'n_regions': len(regional_data.output_gaps.index),
            'time_period': {
                'start': str(regional_data.output_gaps.columns[0]),
                'end': str(regional_data.output_gaps.columns[-1]),
                'n_periods': len(regional_data.output_gaps.columns)
            },
            'data_coverage': {
                'output_gaps_missing': regional_data.output_gaps.isnull().sum().sum(),
                'inflation_missing': regional_data.inflation_rates.isnull().sum().sum(),
                'interest_rates_missing': regional_data.interest_rates.isnull().sum()
            }
        }
    
    def _create_parameter_summary(self, regional_params: RegionalParameters) -> Dict[str, Any]:
        """Create summary of parameter estimates for report."""
        return {
            'sigma_range': {
                'min': float(regional_params.sigma.min()),
                'max': float(regional_params.sigma.max()),
                'mean': float(regional_params.sigma.mean())
            },
            'kappa_range': {
                'min': float(regional_params.kappa.min()),
                'max': float(regional_params.kappa.max()),
                'mean': float(regional_params.kappa.mean())
            },
            'heterogeneity_measures': {
                'sigma_cv': float(regional_params.sigma.std() / regional_params.sigma.mean()),
                'kappa_cv': float(regional_params.kappa.std() / regional_params.kappa.mean())
            }
        }
    
    def _create_policy_summary(self, policy_analysis: PolicyMistakeComponents) -> Dict[str, Any]:
        """Create summary of policy analysis for report."""
        total_abs = abs(policy_analysis.total_mistake)
        
        return {
            'total_mistake': float(policy_analysis.total_mistake),
            'largest_component': self._find_largest_component(policy_analysis),
            'relative_contributions': {
                'information': abs(policy_analysis.information_effect) / total_abs if total_abs > 0 else 0,
                'weight_misallocation': abs(policy_analysis.weight_misallocation_effect) / total_abs if total_abs > 0 else 0,
                'parameter_misspec': abs(policy_analysis.parameter_misspecification_effect) / total_abs if total_abs > 0 else 0,
                'inflation_response': abs(policy_analysis.inflation_response_effect) / total_abs if total_abs > 0 else 0
            }
        }
    
    def _create_counterfactual_summary(self, scenarios: List[PolicyScenario]) -> Dict[str, Any]:
        """Create summary of counterfactual results for report."""
        welfare_outcomes = {scenario.name: scenario.welfare_outcome for scenario in scenarios}
        
        # Find best and worst scenarios
        best_scenario = max(welfare_outcomes, key=welfare_outcomes.get)
        worst_scenario = min(welfare_outcomes, key=welfare_outcomes.get)
        
        return {
            'n_scenarios': len(scenarios),
            'welfare_outcomes': welfare_outcomes,
            'best_scenario': best_scenario,
            'worst_scenario': worst_scenario,
            'welfare_range': welfare_outcomes[best_scenario] - welfare_outcomes[worst_scenario]
        }
    
    def _find_largest_component(self, policy_analysis: PolicyMistakeComponents) -> str:
        """Find the largest component in policy mistake decomposition."""
        components = {
            'Information Effect': abs(policy_analysis.information_effect),
            'Weight Misallocation': abs(policy_analysis.weight_misallocation_effect),
            'Parameter Misspecification': abs(policy_analysis.parameter_misspecification_effect),
            'Inflation Response': abs(policy_analysis.inflation_response_effect)
        }
        
        return max(components, key=components.get)
    
    def _extract_main_conclusions(
        self, 
        key_findings: Dict[str, Any], 
        welfare_gains: Dict[str, float]
    ) -> List[str]:
        """Extract main conclusions for executive summary."""
        conclusions = []
        
        # Add welfare-based conclusions
        if welfare_gains:
            max_gain = max(welfare_gains.values())
            best_scenario = max(welfare_gains, key=welfare_gains.get)
            conclusions.append(
                f"The {best_scenario} scenario offers the highest welfare gains "
                f"with an improvement of {max_gain:.4f} over the baseline."
            )
        
        # Add parameter heterogeneity conclusions
        if 'parameter_heterogeneity' in key_findings:
            conclusions.append(
                "Significant regional heterogeneity in monetary policy transmission "
                "suggests the need for region-specific policy considerations."
            )
        
        return conclusions
    
    def _get_theoretical_framework_text(self) -> str:
        """Get theoretical framework description."""
        return """
        The analysis is based on a multi-region New Keynesian DSGE model with spatial
        spillovers. The model incorporates regional heterogeneity in structural parameters
        and allows for spillover effects through trade, migration, and financial linkages.
        The theoretical framework follows the approach outlined in the mathematical appendix.
        """
    
    def _get_estimation_procedure_text(self) -> str:
        """Get estimation procedure description."""
        return """
        Parameter estimation follows a three-stage procedure: (1) spatial weight matrix
        construction using trade, migration, financial, and distance data; (2) regional
        structural parameter estimation using GMM with appropriate moment conditions;
        (3) policy parameter estimation accounting for Fed's information set and
        reaction function.
        """
    
    def _get_identification_strategy_text(self) -> str:
        """Get identification strategy description."""
        return """
        Identification relies on regional variation in economic conditions and
        spatial spillover patterns. The model is identified through exclusion
        restrictions and the assumption that monetary policy responds to aggregate
        rather than region-specific conditions.
        """
    
    def _create_basic_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create basic HTML report when templates are not available."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .summary-box {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #007bff; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{report_data['title']}</h1>
            <p><strong>Generated:</strong> {report_data['generation_date']}</p>
            
            <div class="summary-box">
                <h2>Data Summary</h2>
                <p><strong>Number of Regions:</strong> {report_data['data_summary']['n_regions']}</p>
                <p><strong>Time Period:</strong> {report_data['data_summary']['time_period']['start']} to {report_data['data_summary']['time_period']['end']}</p>
                <p><strong>Number of Periods:</strong> {report_data['data_summary']['time_period']['n_periods']}</p>
            </div>
            
            <div class="summary-box">
                <h2>Parameter Estimates Summary</h2>
                <p><strong>Interest Rate Sensitivity (σ) Range:</strong> {report_data['parameter_summary']['sigma_range']['min']:.4f} to {report_data['parameter_summary']['sigma_range']['max']:.4f}</p>
                <p><strong>Phillips Curve Slope (κ) Range:</strong> {report_data['parameter_summary']['kappa_range']['min']:.4f} to {report_data['parameter_summary']['kappa_range']['max']:.4f}</p>
            </div>
            
            <div class="summary-box">
                <h2>Policy Analysis Summary</h2>
                <p><strong>Total Policy Mistake:</strong> {report_data['policy_analysis_summary']['total_mistake']:.4f}</p>
                <p><strong>Largest Component:</strong> {report_data['policy_analysis_summary']['largest_component']}</p>
            </div>
            
            <div class="summary-box">
                <h2>Counterfactual Analysis Summary</h2>
                <p><strong>Number of Scenarios:</strong> {report_data['counterfactual_summary']['n_scenarios']}</p>
                <p><strong>Best Scenario:</strong> {report_data['counterfactual_summary']['best_scenario']}</p>
                <p><strong>Welfare Range:</strong> {report_data['counterfactual_summary']['welfare_range']:.6f}</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_basic_methodology_html(self, methodology_data: Dict[str, Any]) -> str:
        """Create basic HTML methodology report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{methodology_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .method-section {{ margin: 30px 0; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{methodology_data['title']}</h1>
            <p><strong>Generated:</strong> {methodology_data['generation_date']}</p>
            
            <div class="method-section">
                <h2>Theoretical Framework</h2>
                <p>{methodology_data['theoretical_framework']}</p>
            </div>
            
            <div class="method-section">
                <h2>Estimation Procedure</h2>
                <p>{methodology_data['estimation_procedure']}</p>
            </div>
            
            <div class="method-section">
                <h2>Identification Strategy</h2>
                <p>{methodology_data['identification_strategy']}</p>
            </div>
            
            <div class="method-section">
                <h2>Model Specification</h2>
                <pre>{json.dumps(methodology_data['model_specification'], indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_basic_summary_html(self, summary_data: Dict[str, Any]) -> str:
        """Create basic HTML executive summary."""
        conclusions_html = "".join([f"<li>{conclusion}</li>" for conclusion in summary_data['main_conclusions']])
        implications_html = "".join([f"<li>{implication}</li>" for implication in summary_data['policy_implications']])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{summary_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .highlight {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{summary_data['title']}</h1>
            <p><strong>Generated:</strong> {summary_data['generation_date']}</p>
            
            <div class="highlight">
                <h2>Main Conclusions</h2>
                <ul>{conclusions_html}</ul>
            </div>
            
            <div class="highlight">
                <h2>Policy Implications</h2>
                <ul>{implications_html}</ul>
            </div>
            
            <h2>Welfare Gains by Scenario</h2>
            <table>
                <tr><th>Scenario</th><th>Welfare Gain</th></tr>
        """
        
        for scenario, gain in summary_data['welfare_gains'].items():
            html += f"<tr><td>{scenario}</td><td>{gain:.6f}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        return html