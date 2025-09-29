"""
Tests for export and report generation functionality.

This module tests the comprehensive export, reporting, and documentation
capabilities of the regional monetary policy analysis framework.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

from regional_monetary_policy.presentation.report_generator import (
    DataExporter, ChartExporter, ReportGenerator
)
from regional_monetary_policy.presentation.documentation import (
    MetadataGenerator, DocumentationGenerator
)
from regional_monetary_policy.presentation.api import RegionalMonetaryPolicyAPI
from regional_monetary_policy.data.models import RegionalDataset
from regional_monetary_policy.econometric.models import RegionalParameters
from regional_monetary_policy.policy.models import PolicyMistakeComponents, PolicyScenario


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_regional_data():
    """Create sample regional dataset for testing."""
    regions = ['CA', 'TX', 'NY']
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='M')
    
    output_gaps = pd.DataFrame(
        np.random.normal(0, 1, (len(regions), len(dates))),
        index=regions,
        columns=dates
    )
    
    inflation_rates = pd.DataFrame(
        np.random.normal(2, 0.5, (len(regions), len(dates))),
        index=regions,
        columns=dates
    )
    
    interest_rates = pd.Series(
        np.random.normal(3, 1, len(dates)),
        index=dates
    )
    
    return RegionalDataset(
        output_gaps=output_gaps,
        inflation_rates=inflation_rates,
        interest_rates=interest_rates,
        real_time_estimates={},
        metadata={'source': 'test_data'}
    )


@pytest.fixture
def sample_regional_parameters():
    """Create sample regional parameters for testing."""
    return RegionalParameters(
        sigma=np.array([0.5, 0.7, 0.6]),
        kappa=np.array([0.1, 0.15, 0.12]),
        psi=np.array([0.2, 0.25, 0.22]),
        phi=np.array([0.3, 0.35, 0.32]),
        beta=np.array([0.99, 0.99, 0.99]),
        standard_errors={
            'sigma': np.array([0.05, 0.07, 0.06]),
            'kappa': np.array([0.02, 0.03, 0.025])
        },
        confidence_intervals={
            'sigma': (np.array([0.4, 0.56, 0.48]), np.array([0.6, 0.84, 0.72]))
        }
    )


@pytest.fixture
def sample_policy_analysis():
    """Create sample policy analysis for testing."""
    return PolicyMistakeComponents(
        total_mistake=0.25,
        information_effect=0.10,
        weight_misallocation_effect=0.08,
        parameter_misspecification_effect=0.05,
        inflation_response_effect=0.02
    )


@pytest.fixture
def sample_counterfactual_results():
    """Create sample counterfactual results for testing."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='M')
    
    return [
        PolicyScenario(
            name='Baseline',
            policy_rates=pd.Series(np.random.normal(3, 1, len(dates)), index=dates),
            regional_outcomes=pd.DataFrame(
                np.random.normal(0, 1, (6, len(dates))),
                columns=dates
            ),
            welfare_outcome=-0.150,
            scenario_type='baseline'
        ),
        PolicyScenario(
            name='Perfect Information',
            policy_rates=pd.Series(np.random.normal(2.8, 0.9, len(dates)), index=dates),
            regional_outcomes=pd.DataFrame(
                np.random.normal(0, 0.9, (6, len(dates))),
                columns=dates
            ),
            welfare_outcome=-0.125,
            scenario_type='perfect_info'
        )
    ]


class TestDataExporter:
    """Test data export functionality."""
    
    def test_export_regional_data_csv(self, temp_output_dir, sample_regional_data):
        """Test CSV export of regional data."""
        exporter = DataExporter(temp_output_dir)
        
        exported_files = exporter.export_regional_data(
            sample_regional_data,
            formats=['csv']
        )
        
        assert 'csv' in exported_files
        assert isinstance(exported_files['csv'], dict)
        
        # Check that files were created
        for file_path in exported_files['csv'].values():
            assert Path(file_path).exists()
    
    def test_export_regional_data_json(self, temp_output_dir, sample_regional_data):
        """Test JSON export of regional data."""
        exporter = DataExporter(temp_output_dir)
        
        exported_files = exporter.export_regional_data(
            sample_regional_data,
            formats=['json']
        )
        
        assert 'json' in exported_files
        assert Path(exported_files['json']).exists()
        
        # Verify JSON content
        with open(exported_files['json'], 'r') as f:
            data = json.load(f)
        
        assert 'output_gaps' in data
        assert 'inflation_rates' in data
        assert 'metadata' in data
    
    def test_export_parameter_estimates(self, temp_output_dir, sample_regional_parameters):
        """Test parameter estimates export."""
        exporter = DataExporter(temp_output_dir)
        
        exported_files = exporter.export_parameter_estimates(
            sample_regional_parameters,
            formats=['csv', 'json']
        )
        
        assert 'csv' in exported_files
        assert 'json' in exported_files
        
        # Check CSV file
        csv_data = pd.read_csv(exported_files['csv'])
        assert len(csv_data) == 3  # 3 regions
        
        # Check JSON file
        with open(exported_files['json'], 'r') as f:
            json_data = json.load(f)
        
        assert 'sigma' in json_data
        assert len(json_data['sigma']) == 3
    
    def test_export_policy_analysis(self, temp_output_dir, sample_policy_analysis):
        """Test policy analysis export."""
        exporter = DataExporter(temp_output_dir)
        
        exported_files = exporter.export_policy_analysis(
            sample_policy_analysis,
            formats=['csv', 'json']
        )
        
        assert 'csv' in exported_files
        assert 'json' in exported_files
        
        # Check CSV content
        csv_data = pd.read_csv(exported_files['csv'])
        assert 'Component' in csv_data.columns
        assert 'Value' in csv_data.columns
        assert len(csv_data) == 5  # 5 components
    
    def test_export_counterfactual_results(self, temp_output_dir, sample_counterfactual_results):
        """Test counterfactual results export."""
        exporter = DataExporter(temp_output_dir)
        
        from regional_monetary_policy.policy.models import ComparisonResults
        comparison_results = ComparisonResults(sample_counterfactual_results)
        
        exported_files = exporter.export_counterfactual_results(
            sample_counterfactual_results,
            comparison_results,
            formats=['csv', 'json']
        )
        
        assert 'csv' in exported_files
        assert 'json' in exported_files


class TestChartExporter:
    """Test chart export functionality."""
    
    def test_export_figure_png(self, temp_output_dir):
        """Test PNG export of figures."""
        # Skip if plotly not available for image export
        pytest.importorskip("kaleido")
        
        import plotly.graph_objects as go
        
        exporter = ChartExporter(temp_output_dir)
        
        # Create simple figure
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        fig.update_layout(title="Test Figure")
        
        exported_files = exporter.export_figure(
            fig,
            "test_chart",
            formats=['png']
        )
        
        assert 'png' in exported_files
        assert Path(exported_files['png']).exists()
    
    def test_export_multiple_figures(self, temp_output_dir):
        """Test export of multiple figures."""
        # Skip if plotly not available for image export
        pytest.importorskip("kaleido")
        
        import plotly.graph_objects as go
        
        exporter = ChartExporter(temp_output_dir)
        
        figures = {
            'chart1': go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4])),
            'chart2': go.Figure(data=go.Bar(x=['A', 'B'], y=[1, 2]))
        }
        
        exported_files = exporter.export_multiple_figures(
            figures,
            formats=['png']
        )
        
        assert 'chart1' in exported_files
        assert 'chart2' in exported_files


class TestReportGenerator:
    """Test report generation functionality."""
    
    def test_generate_comprehensive_report(
        self, 
        temp_output_dir, 
        sample_regional_data,
        sample_regional_parameters,
        sample_policy_analysis,
        sample_counterfactual_results
    ):
        """Test comprehensive report generation."""
        generator = ReportGenerator(temp_output_dir)
        
        report_path = generator.generate_comprehensive_report(
            sample_regional_data,
            sample_regional_parameters,
            sample_policy_analysis,
            sample_counterfactual_results
        )
        
        assert Path(report_path).exists()
        
        # Check that it's an HTML file
        assert report_path.endswith('.html')
        
        # Check basic content
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert 'Regional Monetary Policy Analysis Report' in content
        assert 'Data Summary' in content
    
    def test_generate_methodology_report(self, temp_output_dir):
        """Test methodology report generation."""
        generator = ReportGenerator(temp_output_dir)
        
        estimation_config = {'method': 'GMM', 'tolerance': 1e-6}
        model_spec = {'framework': 'New Keynesian', 'regions': ['CA', 'TX']}
        
        report_path = generator.generate_methodology_report(
            estimation_config,
            model_spec
        )
        
        assert Path(report_path).exists()
        assert report_path.endswith('.html')
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert 'Methodology' in content
        assert 'Theoretical Framework' in content
    
    def test_generate_executive_summary(self, temp_output_dir):
        """Test executive summary generation."""
        generator = ReportGenerator(temp_output_dir)
        
        key_findings = {'heterogeneity': 'significant'}
        policy_implications = ['Need for regional considerations']
        welfare_gains = {'Perfect Regional': 0.05, 'Baseline': 0.0}
        
        summary_path = generator.generate_executive_summary(
            key_findings,
            policy_implications,
            welfare_gains
        )
        
        assert Path(summary_path).exists()
        assert summary_path.endswith('.html')


class TestMetadataGenerator:
    """Test metadata generation functionality."""
    
    def test_generate_data_metadata(self, temp_output_dir, sample_regional_data):
        """Test data metadata generation."""
        generator = MetadataGenerator(temp_output_dir)
        
        metadata = generator.generate_data_metadata(sample_regional_data)
        
        assert 'dataset_info' in metadata
        assert 'variables' in metadata
        assert 'data_quality' in metadata
        
        # Check dataset info
        dataset_info = metadata['dataset_info']
        assert dataset_info['n_regions'] == 3
        assert len(dataset_info['regions']) == 3
    
    def test_generate_estimation_metadata(self, temp_output_dir, sample_regional_parameters):
        """Test estimation metadata generation."""
        generator = MetadataGenerator(temp_output_dir)
        
        estimation_config = {'method': 'GMM', 'tolerance': 1e-6}
        
        metadata = generator.generate_estimation_metadata(
            sample_regional_parameters,
            estimation_config
        )
        
        assert 'estimation_info' in metadata
        assert 'parameter_estimates' in metadata
        assert 'model_specification' in metadata
        
        # Check parameter estimates
        param_estimates = metadata['parameter_estimates']
        assert 'sigma' in param_estimates
        assert len(param_estimates['sigma']['estimates']) == 3
    
    def test_generate_policy_analysis_metadata(self, temp_output_dir, sample_policy_analysis):
        """Test policy analysis metadata generation."""
        generator = MetadataGenerator(temp_output_dir)
        
        metadata = generator.generate_policy_analysis_metadata(sample_policy_analysis)
        
        assert 'analysis_info' in metadata
        assert 'mistake_components' in metadata
        assert 'relative_contributions' in metadata
        
        # Check mistake components
        components = metadata['mistake_components']
        assert 'total_mistake' in components
        assert components['total_mistake']['value'] == 0.25
    
    def test_save_metadata_json(self, temp_output_dir):
        """Test saving metadata in JSON format."""
        generator = MetadataGenerator(temp_output_dir)
        
        test_metadata = {'test': 'data', 'number': 42}
        
        filepath = generator.save_metadata(test_metadata, 'test_metadata', 'json')
        
        assert Path(filepath).exists()
        assert filepath.endswith('.json')
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['test'] == 'data'
        assert loaded_data['number'] == 42
    
    def test_generate_complete_metadata(
        self,
        temp_output_dir,
        sample_regional_data,
        sample_regional_parameters,
        sample_policy_analysis,
        sample_counterfactual_results
    ):
        """Test complete metadata generation."""
        generator = MetadataGenerator(temp_output_dir)
        
        metadata = generator.generate_complete_metadata(
            sample_regional_data,
            sample_regional_parameters,
            sample_policy_analysis,
            sample_counterfactual_results
        )
        
        assert 'analysis_overview' in metadata
        assert 'data_metadata' in metadata
        assert 'estimation_metadata' in metadata
        assert 'policy_analysis_metadata' in metadata
        assert 'counterfactual_metadata' in metadata
        assert 'reproducibility_info' in metadata


class TestDocumentationGenerator:
    """Test documentation generation functionality."""
    
    def test_generate_user_guide(self, temp_output_dir):
        """Test user guide generation."""
        generator = DocumentationGenerator(temp_output_dir)
        
        guide_path = generator.generate_user_guide()
        
        assert Path(guide_path).exists()
        assert guide_path.endswith('.md')
        
        with open(guide_path, 'r') as f:
            content = f.read()
        
        assert 'User Guide' in content
        assert 'Getting Started' in content
        assert 'Installation' in content
    
    def test_generate_api_documentation(self, temp_output_dir):
        """Test API documentation generation."""
        generator = DocumentationGenerator(temp_output_dir)
        
        api_docs_path = generator.generate_api_documentation()
        
        assert Path(api_docs_path).exists()
        assert api_docs_path.endswith('.md')
        
        with open(api_docs_path, 'r') as f:
            content = f.read()
        
        assert 'API Documentation' in content
        assert 'RegionalMonetaryPolicyAPI' in content
    
    def test_generate_methodology_documentation(self, temp_output_dir):
        """Test methodology documentation generation."""
        generator = DocumentationGenerator(temp_output_dir)
        
        methodology_path = generator.generate_methodology_documentation()
        
        assert Path(methodology_path).exists()
        assert methodology_path.endswith('.md')
        
        with open(methodology_path, 'r') as f:
            content = f.read()
        
        assert 'Methodology Documentation' in content
        assert 'Theoretical Framework' in content
        assert 'Estimation Methodology' in content


class TestIntegration:
    """Test integration of export and reporting components."""
    
    def test_full_export_workflow(
        self,
        temp_output_dir,
        sample_regional_data,
        sample_regional_parameters,
        sample_policy_analysis,
        sample_counterfactual_results
    ):
        """Test complete export workflow."""
        # Initialize all components
        data_exporter = DataExporter(str(Path(temp_output_dir) / "exports"))
        report_generator = ReportGenerator(str(Path(temp_output_dir) / "reports"))
        metadata_generator = MetadataGenerator(str(Path(temp_output_dir) / "metadata"))
        
        # Export data
        data_exports = data_exporter.export_regional_data(
            sample_regional_data,
            formats=['csv', 'json']
        )
        
        # Generate reports
        comprehensive_report = report_generator.generate_comprehensive_report(
            sample_regional_data,
            sample_regional_parameters,
            sample_policy_analysis,
            sample_counterfactual_results
        )
        
        # Generate metadata
        complete_metadata = metadata_generator.generate_complete_metadata(
            sample_regional_data,
            sample_regional_parameters,
            sample_policy_analysis,
            sample_counterfactual_results
        )
        
        # Verify all outputs exist
        assert data_exports is not None
        assert Path(comprehensive_report).exists()
        assert complete_metadata is not None
        
        # Check directory structure
        exports_dir = Path(temp_output_dir) / "exports"
        reports_dir = Path(temp_output_dir) / "reports"
        metadata_dir = Path(temp_output_dir) / "metadata"
        
        assert exports_dir.exists()
        assert reports_dir.exists()
        assert metadata_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])