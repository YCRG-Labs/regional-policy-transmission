"""
Presentation module for regional monetary policy analysis.

This module handles visualization, reporting, export functionality, and 
programmatic access for displaying analysis results and interactive exploration.
"""

from .visualizers import (
    RegionalMapVisualizer,
    TimeSeriesVisualizer,
    ParameterVisualizer,
    PolicyAnalysisVisualizer,
    CounterfactualVisualizer
)

from .report_generator import (
    ReportGenerator,
    DataExporter,
    ChartExporter
)

from .api import (
    RegionalMonetaryPolicyAPI,
    AnalysisWorkflow
)

from .documentation import (
    MetadataGenerator,
    DocumentationGenerator
)

__all__ = [
    # Visualization components
    'RegionalMapVisualizer',
    'TimeSeriesVisualizer', 
    'ParameterVisualizer',
    'PolicyAnalysisVisualizer',
    'CounterfactualVisualizer',
    
    # Export and reporting components
    'ReportGenerator',
    'DataExporter',
    'ChartExporter',
    
    # API and workflow components
    'RegionalMonetaryPolicyAPI',
    'AnalysisWorkflow',
    
    # Documentation and metadata components
    'MetadataGenerator',
    'DocumentationGenerator'
]