"""
Inline Help and Guidance System for Regional Monetary Policy Analysis Web Interface.

Provides contextual help, tooltips, tutorials, and guidance throughout
the web application to assist users in conducting regional monetary policy analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json


class HelpSystem:
    """
    Comprehensive help system for the web interface.
    
    Provides contextual help, tooltips, guided tutorials, and troubleshooting
    assistance throughout the regional monetary policy analysis workflow.
    """
    
    def __init__(self):
        """Initialize the help system with content and configuration."""
        self.help_content = self._load_help_content()
        self.tutorial_steps = self._load_tutorial_steps()
        self.tooltips = self._load_tooltips()
        self.troubleshooting = self._load_troubleshooting_guide()
    
    def _load_help_content(self) -> Dict[str, Any]:
        """Load help content for different sections of the application."""
        return {
            'data_loading': {
                'title': 'Data Loading and Validation',
                'description': 'Load regional economic data from FRED API and validate quality',
                'steps': [
                    'Enter your FRED API key',
                    'Select regions for analysis',
                    'Choose time period and data frequency',
                    'Review data quality metrics',
                    'Handle missing values if needed'
                ],
                'tips': [
                    'Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html',
                    'Start with 4-6 regions for initial analysis',
                    'Use at least 10 years of data for reliable estimation',
                    'Check for structural breaks in your data period'
                ]
            },
            'parameter_estimation': {
                'title': 'Regional Parameter Estimation',
                'description': 'Estimate structural parameters using three-stage GMM procedure',
                'steps': [
                    'Configure spatial weight matrix',
                    'Set estimation options (GMM settings)',
                    'Run identification tests',
                    'Estimate regional parameters',
                    'Review estimation diagnostics'
                ],
                'tips': [
                    'Ensure spatial weights are row-normalized',
                    'Check identification conditions before estimation',
                    'Use robust standard errors for inference',
                    'Compare results with literature benchmarks'
                ]
            },
            'policy_analysis': {
                'title': 'Policy Mistake Decomposition',
                'description': 'Decompose Fed policy mistakes into constituent components',
                'steps': [
                    'Estimate Fed reaction function',
                    'Calculate optimal policy parameters',
                    'Reconstruct Fed information sets',
                    'Perform mistake decomposition',
                    'Interpret component contributions'
                ],
                'tips': [
                    'Use real-time data for Fed information sets',
                    'Consider parameter uncertainty in decomposition',
                    'Focus on economically significant components',
                    'Validate results across different periods'
                ]
            },
            'counterfactual_analysis': {
                'title': 'Counterfactual Policy Analysis',
                'description': 'Generate and compare alternative policy scenarios',
                'steps': [
                    'Generate baseline scenario (historical Fed policy)',
                    'Create perfect information scenario',
                    'Generate optimal regional scenarios',
                    'Compute welfare outcomes',
                    'Compare scenario rankings'
                ],
                'tips': [
                    'Verify welfare ranking: W^PR ≥ W^PI ≥ W^OR ≥ W^B',
                    'Check scenario plausibility',
                    'Quantify welfare gains from policy improvements',
                    'Analyze regional distributional effects'
                ]
            },
            'visualization': {
                'title': 'Results Visualization',
                'description': 'Create interactive charts and maps for analysis results',
                'steps': [
                    'Select visualization type',
                    'Configure display options',
                    'Customize colors and labels',
                    'Export high-resolution figures',
                    'Generate publication-ready tables'
                ],
                'tips': [
                    'Use consistent color schemes across charts',
                    'Include confidence intervals where appropriate',
                    'Add clear titles and axis labels',
                    'Consider accessibility in color choices'
                ]
            }
        }
    
    def _load_tutorial_steps(self) -> Dict[str, List[Dict]]:
        """Load guided tutorial steps for each analysis workflow."""
        return {
            'getting_started': [
                {
                    'step': 1,
                    'title': 'Setup API Access',
                    'description': 'Configure FRED API key for data access',
                    'action': 'Enter your FRED API key in the sidebar',
                    'validation': 'API key should be 32 characters long'
                },
                {
                    'step': 2,
                    'title': 'Select Regions',
                    'description': 'Choose regions for your analysis',
                    'action': 'Select 4-6 regions from the dropdown menu',
                    'validation': 'At least 3 regions required for spatial analysis'
                },
                {
                    'step': 3,
                    'title': 'Load Data',
                    'description': 'Retrieve regional economic data',
                    'action': 'Click "Load Data" and wait for completion',
                    'validation': 'Data quality score should be > 7.0'
                },
                {
                    'step': 4,
                    'title': 'Review Data Quality',
                    'description': 'Check data validation results',
                    'action': 'Review the data quality report',
                    'validation': 'Address any quality issues before proceeding'
                }
            ],
            'full_analysis': [
                {
                    'step': 1,
                    'title': 'Data Preparation',
                    'description': 'Load and validate regional data',
                    'action': 'Complete data loading workflow',
                    'validation': 'Data loaded successfully with good quality'
                },
                {
                    'step': 2,
                    'title': 'Parameter Estimation',
                    'description': 'Estimate regional structural parameters',
                    'action': 'Run three-stage GMM estimation',
                    'validation': 'Parameters estimated with reasonable standard errors'
                },
                {
                    'step': 3,
                    'title': 'Policy Analysis',
                    'description': 'Analyze Fed policy decisions',
                    'action': 'Perform policy mistake decomposition',
                    'validation': 'Decomposition components sum to total mistake'
                },
                {
                    'step': 4,
                    'title': 'Counterfactual Scenarios',
                    'description': 'Generate alternative policy scenarios',
                    'action': 'Create and compare counterfactual scenarios',
                    'validation': 'Welfare ranking follows theoretical predictions'
                },
                {
                    'step': 5,
                    'title': 'Results Interpretation',
                    'description': 'Interpret and visualize results',
                    'action': 'Generate charts and summary reports',
                    'validation': 'Results are economically meaningful'
                }
            ]
        }
    
    def _load_tooltips(self) -> Dict[str, str]:
        """Load tooltip text for UI elements."""
        return {
            'fred_api_key': 'Your FRED API key for accessing economic data. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html',
            'regions_selection': 'Select U.S. states or regions for analysis. More regions provide richer spatial analysis but increase computation time.',
            'time_period': 'Analysis time period. Use at least 10 years of data for reliable parameter estimation.',
            'data_frequency': 'Monthly data provides more observations but may be noisier. Quarterly data is more stable but fewer observations.',
            'spatial_weights': 'How regions influence each other. Combined weights use trade, migration, financial, and distance data.',
            'gmm_weighting': 'Optimal weighting matrix provides efficient estimates. Identity matrix is more robust to misspecification.',
            'identification_strategy': 'Method for identifying structural parameters. External instruments help with endogeneity concerns.',
            'welfare_weights': 'How to weight different regions in social welfare function. Population-based weights reflect democratic principles.',
            'confidence_level': 'Confidence level for statistical inference. 95% is standard in academic research.',
            'bootstrap_replications': 'Number of bootstrap samples for uncertainty quantification. More replications give more precise confidence intervals.',
            'visualization_theme': 'Color scheme for charts and maps. Choose based on your presentation needs and accessibility requirements.'
        }
    
    def _load_troubleshooting_guide(self) -> Dict[str, Dict]:
        """Load troubleshooting guide for common issues."""
        return {
            'api_connection_failed': {
                'problem': 'Cannot connect to FRED API',
                'causes': [
                    'Invalid or expired API key',
                    'Network connectivity issues',
                    'FRED API service temporarily unavailable'
                ],
                'solutions': [
                    'Verify your API key is correct and active',
                    'Check your internet connection',
                    'Try again in a few minutes',
                    'Contact FRED support if problem persists'
                ]
            },
            'estimation_convergence_failed': {
                'problem': 'Parameter estimation does not converge',
                'causes': [
                    'Insufficient data for identification',
                    'Poor spatial weight specification',
                    'Numerical optimization issues'
                ],
                'solutions': [
                    'Increase maximum iterations',
                    'Try different starting values',
                    'Use robust optimization algorithms',
                    'Check spatial weight matrix properties'
                ]
            },
            'data_quality_poor': {
                'problem': 'Data quality score is low',
                'causes': [
                    'Many missing observations',
                    'Outliers in the data',
                    'Structural breaks in time series'
                ],
                'solutions': [
                    'Use data interpolation for missing values',
                    'Apply outlier detection and treatment',
                    'Consider shorter time periods',
                    'Use alternative data sources'
                ]
            },
            'welfare_ranking_violated': {
                'problem': 'Counterfactual welfare ranking is incorrect',
                'causes': [
                    'Estimation uncertainty',
                    'Model misspecification',
                    'Numerical precision issues'
                ],
                'solutions': [
                    'Check parameter estimation quality',
                    'Use bootstrap confidence intervals',
                    'Try alternative model specifications',
                    'Increase numerical precision'
                ]
            }
        }
    
    def show_contextual_help(self, section: str) -> None:
        """Display contextual help for a specific section."""
        if section in self.help_content:
            content = self.help_content[section]
            
            with st.expander(f"ℹ️ Help: {content['title']}", expanded=False):
                st.write(content['description'])
                
                st.subheader("Steps:")
                for i, step in enumerate(content['steps'], 1):
                    st.write(f"{i}. {step}")
                
                st.subheader("Tips:")
                for tip in content['tips']:
                    st.write(f"💡 {tip}")
    
    def show_tooltip(self, element_key: str) -> str:
        """Get tooltip text for a UI element."""
        return self.tooltips.get(element_key, "No help available for this element.")
    
    def show_guided_tutorial(self, tutorial_name: str, current_step: int = 1) -> None:
        """Display guided tutorial with step-by-step instructions."""
        if tutorial_name in self.tutorial_steps:
            steps = self.tutorial_steps[tutorial_name]
            
            st.sidebar.markdown("## 📚 Guided Tutorial")
            
            # Progress indicator
            progress = current_step / len(steps)
            st.sidebar.progress(progress)
            st.sidebar.write(f"Step {current_step} of {len(steps)}")
            
            # Current step details
            if current_step <= len(steps):
                step = steps[current_step - 1]
                
                st.sidebar.markdown(f"### {step['title']}")
                st.sidebar.write(step['description'])
                st.sidebar.markdown(f"**Action:** {step['action']}")
                st.sidebar.markdown(f"**Validation:** {step['validation']}")
                
                # Navigation buttons
                col1, col2 = st.sidebar.columns(2)
                
                if current_step > 1:
                    if col1.button("← Previous"):
                        st.session_state.tutorial_step = current_step - 1
                        st.experimental_rerun()
                
                if current_step < len(steps):
                    if col2.button("Next →"):
                        st.session_state.tutorial_step = current_step + 1
                        st.experimental_rerun()
                else:
                    if col2.button("Complete ✓"):
                        st.session_state.tutorial_active = False
                        st.experimental_rerun()
    
    def show_troubleshooting(self, issue_key: Optional[str] = None) -> None:
        """Display troubleshooting guide."""
        st.subheader("🔧 Troubleshooting Guide")
        
        if issue_key and issue_key in self.troubleshooting:
            # Show specific issue
            issue = self.troubleshooting[issue_key]
            
            st.error(f"**Problem:** {issue['problem']}")
            
            st.write("**Possible Causes:**")
            for cause in issue['causes']:
                st.write(f"• {cause}")
            
            st.write("**Solutions:**")
            for solution in issue['solutions']:
                st.write(f"✓ {solution}")
        
        else:
            # Show all issues
            for key, issue in self.troubleshooting.items():
                with st.expander(issue['problem']):
                    st.write("**Possible Causes:**")
                    for cause in issue['causes']:
                        st.write(f"• {cause}")
                    
                    st.write("**Solutions:**")
                    for solution in issue['solutions']:
                        st.write(f"✓ {solution}")
    
    def show_quick_help_panel(self) -> None:
        """Display quick help panel in sidebar."""
        with st.sidebar.expander("❓ Quick Help", expanded=False):
            st.markdown("""
            **Getting Started:**
            1. Enter FRED API key
            2. Select regions
            3. Load data
            4. Run analysis
            
            **Need Help?**
            - Click ℹ️ icons for contextual help
            - Use guided tutorial mode
            - Check troubleshooting guide
            
            **Keyboard Shortcuts:**
            - Ctrl+Enter: Run analysis
            - Ctrl+S: Save results
            - F1: Show help
            """)
    
    def create_help_button(self, section: str, label: str = "Help") -> None:
        """Create a help button that shows contextual help when clicked."""
        if st.button(f"ℹ️ {label}", key=f"help_{section}"):
            self.show_contextual_help(section)
    
    def add_tooltip_to_element(self, element_key: str, element) -> None:
        """Add tooltip to a Streamlit element."""
        tooltip_text = self.show_tooltip(element_key)
        element.help = tooltip_text
    
    def check_analysis_readiness(self, session_state) -> Dict[str, bool]:
        """Check if analysis is ready to proceed and provide guidance."""
        readiness = {
            'api_key_valid': hasattr(session_state, 'fred_client') and session_state.fred_client is not None,
            'data_loaded': hasattr(session_state, 'regional_data') and session_state.regional_data is not None,
            'parameters_estimated': hasattr(session_state, 'regional_parameters') and session_state.regional_parameters is not None,
            'spatial_weights_ready': hasattr(session_state, 'spatial_weights') and session_state.spatial_weights is not None
        }
        
        # Provide guidance for missing components
        if not readiness['api_key_valid']:
            st.warning("⚠️ Please enter a valid FRED API key to proceed.")
        
        if not readiness['data_loaded']:
            st.warning("⚠️ Please load regional data before running analysis.")
        
        if not readiness['parameters_estimated']:
            st.info("ℹ️ Run parameter estimation to enable policy analysis.")
        
        return readiness
    
    def show_analysis_summary(self, results: Dict) -> None:
        """Show summary of analysis results with interpretation guidance."""
        st.subheader("📊 Analysis Summary")
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Regions Analyzed", results.get('n_regions', 'N/A'))
        
        with col2:
            st.metric("Time Periods", results.get('n_periods', 'N/A'))
        
        with col3:
            st.metric("Data Quality", f"{results.get('data_quality', 0):.1f}/10")
        
        with col4:
            st.metric("Model Fit (R²)", f"{results.get('r_squared', 0):.3f}")
        
        # Interpretation guidance
        st.markdown("""
        **Interpretation Guide:**
        - **Data Quality > 7.0**: Good quality for reliable analysis
        - **Model Fit > 0.5**: Reasonable explanatory power
        - **Parameter Significance**: Check t-statistics > 2.0
        - **Economic Significance**: Focus on economically meaningful magnitudes
        """)


# Global help system instance
help_system = HelpSystem()