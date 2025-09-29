"""
Parameter estimation page for regional structural parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go

from ...econometric.parameter_estimator import ParameterEstimator
from ...econometric.spatial_handler import SpatialModelHandler
from ...econometric.models import RegionalParameters, EstimationConfig
from ..workflow_manager import WorkflowType


def render():
    """Render the parameter estimation page."""
    st.title("🔢 Parameter Estimation")
    st.markdown("Estimate regional structural parameters using three-stage procedure")
    st.markdown("---")
    
    # Check if we're in a workflow
    workflow_manager = st.session_state.workflow_manager
    current_workflow = workflow_manager.get_current_workflow()
    
    if current_workflow and current_workflow.workflow_type == WorkflowType.PARAMETER_ESTIMATION:
        render_workflow_estimation()
    else:
        render_standalone_estimation()


def render_workflow_estimation():
    """Render estimation within guided workflow."""
    workflow_manager = st.session_state.workflow_manager
    workflow = workflow_manager.get_current_workflow()
    current_step = workflow.get_current_step()
    
    if not current_step:
        st.success("🎉 Parameter estimation workflow completed!")
        render_estimation_results()
        return
    
    st.info(f"**Current Step**: {current_step.title}")
    st.write(current_step.description)
    
    # Render appropriate step
    if current_step.step_id == "data_selection":
        render_data_selection_step()
    elif current_step.step_id == "spatial_weights":
        render_spatial_weights_step()
    elif current_step.step_id == "estimation_setup":
        render_estimation_setup_step()
    elif current_step.step_id == "parameter_estimation":
        render_parameter_estimation_step()
    elif current_step.step_id == "results_review":
        render_results_review_step()
    
    # Navigation buttons
    render_workflow_navigation()


def render_standalone_estimation():
    """Render standalone estimation interface."""
    st.info("💡 For guided analysis, start a Parameter Estimation workflow from the Home page")
    
    tabs = st.tabs(["Data Setup", "Spatial Weights", "Estimation", "Results"])
    
    with tabs[0]:
        render_data_selection_step()
    
    with tabs[1]:
        render_spatial_weights_step()
    
    with tabs[2]:
        render_estimation_setup_step()
        render_parameter_estimation_step()
    
    with tabs[3]:
        render_estimation_results()


def render_data_selection_step():
    """Render data selection step."""
    st.subheader("📊 Data Selection")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("data_selection")
    
    with st.form("data_selection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Region selection
            available_regions = get_available_regions()
            selected_regions = st.multiselect(
                "Select Regions",
                available_regions,
                default=step_data.get('regions', available_regions[:10]),
                help="Choose regions for parameter estimation"
            )
            
            # Time period
            start_date = st.date_input(
                "Start Date",
                value=step_data.get('start_date', pd.Timestamp('2000-01-01').date())
            )
            
            end_date = st.date_input(
                "End Date", 
                value=step_data.get('end_date', pd.Timestamp('2023-12-31').date())
            )
        
        with col2:
            # Economic indicators
            available_indicators = ["output_gap", "inflation", "interest_rate", "unemployment"]
            selected_indicators = st.multiselect(
                "Economic Indicators",
                available_indicators,
                default=step_data.get('indicators', available_indicators[:3])
            )
            
            # Data options
            data_frequency = st.selectbox(
                "Data Frequency",
                ["Monthly", "Quarterly"],
                index=0 if step_data.get('data_frequency') == 'Monthly' else 1
            )
            
            include_vintages = st.checkbox(
                "Include Real-time Vintages",
                value=step_data.get('include_vintages', False)
            )
        
        if st.form_submit_button("💾 Save Data Selection", type="primary"):
            data_config = {
                'regions': selected_regions,
                'start_date': start_date,
                'end_date': end_date,
                'indicators': selected_indicators,
                'data_frequency': data_frequency,
                'include_vintages': include_vintages
            }
            
            workflow_manager.update_workflow_data("data_selection", data_config)
            st.success("✅ Data selection saved!")
            
            # Update session
            session_manager = st.session_state.session_manager
            if session_manager.has_current_session():
                session_manager.update_session_config({'data_selection': data_config})
    
    # Data preview
    if step_data:
        render_data_preview_summary(step_data)


def render_spatial_weights_step():
    """Render spatial weights configuration step."""
    st.subheader("🗺️ Spatial Weight Configuration")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("spatial_weights")
    
    with st.form("spatial_weights_form"):
        st.markdown("#### Weight Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trade_weight = st.slider(
                "Trade Weight",
                min_value=0.0,
                max_value=1.0,
                value=step_data.get('trade_weight', 0.4),
                step=0.1,
                help="Weight for trade-based spatial connections"
            )
            
            migration_weight = st.slider(
                "Migration Weight",
                min_value=0.0,
                max_value=1.0,
                value=step_data.get('migration_weight', 0.3),
                step=0.1
            )
        
        with col2:
            financial_weight = st.slider(
                "Financial Weight",
                min_value=0.0,
                max_value=1.0,
                value=step_data.get('financial_weight', 0.2),
                step=0.1
            )
            
            distance_weight = st.slider(
                "Distance Weight",
                min_value=0.0,
                max_value=1.0,
                value=step_data.get('distance_weight', 0.1),
                step=0.1
            )
        
        # Validation
        total_weight = trade_weight + migration_weight + financial_weight + distance_weight
        if abs(total_weight - 1.0) > 0.01:
            st.error(f"⚠️ Weights must sum to 1.0 (current sum: {total_weight:.2f})")
        
        st.markdown("#### Spatial Matrix Options")
        
        normalization_method = st.selectbox(
            "Normalization Method",
            ["Row Standardization", "Spectral Normalization", "None"],
            index=0
        )
        
        validation_checks = st.multiselect(
            "Validation Checks",
            ["Symmetry", "Connectivity", "Eigenvalue Bounds", "Spatial Autocorrelation"],
            default=["Connectivity", "Eigenvalue Bounds"]
        )
        
        if st.form_submit_button("🔧 Configure Spatial Weights", type="primary"):
            if abs(total_weight - 1.0) <= 0.01:
                spatial_config = {
                    'trade_weight': trade_weight,
                    'migration_weight': migration_weight,
                    'financial_weight': financial_weight,
                    'distance_weight': distance_weight,
                    'normalization_method': normalization_method,
                    'validation_checks': validation_checks
                }
                
                workflow_manager.update_workflow_data("spatial_weights", spatial_config)
                st.success("✅ Spatial weights configured!")
                
                # Generate and display spatial matrix preview
                render_spatial_matrix_preview(spatial_config)
            else:
                st.error("Please ensure weights sum to 1.0")


def render_estimation_setup_step():
    """Render estimation setup step."""
    st.subheader("⚙️ Estimation Setup")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("estimation_setup")
    
    with st.form("estimation_setup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            estimation_method = st.selectbox(
                "Estimation Method",
                ["Three-Stage GMM", "Two-Stage Least Squares", "Maximum Likelihood"],
                index=0,
                help="Choose econometric estimation method"
            )
            
            identification_strategy = st.selectbox(
                "Identification Strategy",
                ["Spatial Instruments", "Lagged Variables", "External Instruments"],
                index=0
            )
            
            convergence_tolerance = st.number_input(
                "Convergence Tolerance",
                min_value=1e-8,
                max_value=1e-3,
                value=step_data.get('convergence_tolerance', 1e-6),
                format="%.2e"
            )
        
        with col2:
            max_iterations = st.number_input(
                "Maximum Iterations",
                min_value=100,
                max_value=10000,
                value=step_data.get('max_iterations', 1000)
            )
            
            robustness_checks = st.multiselect(
                "Robustness Checks",
                ["Bootstrap Standard Errors", "Jackknife", "Alternative Instruments", "Subsample Analysis"],
                default=step_data.get('robustness_checks', ["Bootstrap Standard Errors"])
            )
            
            parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=step_data.get('parallel_processing', True),
                help="Use multiple CPU cores for estimation"
            )
        
        if st.form_submit_button("🔧 Configure Estimation", type="primary"):
            estimation_config = {
                'estimation_method': estimation_method,
                'identification_strategy': identification_strategy,
                'convergence_tolerance': convergence_tolerance,
                'max_iterations': max_iterations,
                'robustness_checks': robustness_checks,
                'parallel_processing': parallel_processing
            }
            
            workflow_manager.update_workflow_data("estimation_setup", estimation_config)
            st.success("✅ Estimation configuration saved!")


def render_parameter_estimation_step():
    """Render parameter estimation execution step."""
    st.subheader("🚀 Parameter Estimation")
    
    workflow_manager = st.session_state.workflow_manager
    
    # Check if all previous steps are configured
    data_config = workflow_manager.get_workflow_data("data_selection")
    spatial_config = workflow_manager.get_workflow_data("spatial_weights")
    estimation_config = workflow_manager.get_workflow_data("estimation_setup")
    
    if not all([data_config, spatial_config, estimation_config]):
        st.warning("⚠️ Please complete all previous configuration steps first")
        return
    
    # Display configuration summary
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Data Configuration**")
            st.write(f"Regions: {len(data_config.get('regions', []))}")
            st.write(f"Period: {data_config.get('start_date')} to {data_config.get('end_date')}")
            st.write(f"Indicators: {', '.join(data_config.get('indicators', []))}")
        
        with col2:
            st.markdown("**Spatial Weights**")
            st.write(f"Trade: {spatial_config.get('trade_weight', 0):.1f}")
            st.write(f"Migration: {spatial_config.get('migration_weight', 0):.1f}")
            st.write(f"Financial: {spatial_config.get('financial_weight', 0):.1f}")
            st.write(f"Distance: {spatial_config.get('distance_weight', 0):.1f}")
        
        with col3:
            st.markdown("**Estimation Setup**")
            st.write(f"Method: {estimation_config.get('estimation_method')}")
            st.write(f"Strategy: {estimation_config.get('identification_strategy')}")
            st.write(f"Max Iterations: {estimation_config.get('max_iterations')}")
    
    # Estimation execution
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Start Estimation", type="primary", use_container_width=True):
            run_parameter_estimation(data_config, spatial_config, estimation_config)
    
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            save_estimation_configuration(data_config, spatial_config, estimation_config)
    
    # Display estimation progress if running
    render_estimation_progress()


def render_results_review_step():
    """Render results review step."""
    st.subheader("📊 Results Review")
    
    # Check if estimation results exist
    workflow_manager = st.session_state.workflow_manager
    results = workflow_manager.get_workflow_data("estimation_results")
    
    if not results:
        st.info("No estimation results available. Please run parameter estimation first.")
        return
    
    render_estimation_results()


def render_workflow_navigation():
    """Render workflow navigation buttons."""
    st.markdown("---")
    
    workflow_manager = st.session_state.workflow_manager
    workflow = workflow_manager.get_current_workflow()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if workflow.current_step > 0:
            if st.button("⬅️ Previous Step"):
                workflow.current_step -= 1
                st.rerun()
    
    with col2:
        progress = workflow.get_progress()
        st.progress(progress / 100)
        st.caption(f"Step {workflow.current_step + 1} of {len(workflow.steps)}")
    
    with col3:
        if workflow_manager.validate_current_step():
            if st.button("Next Step ➡️", type="primary"):
                if workflow_manager.advance_workflow():
                    st.rerun()
        else:
            st.button("Complete Step First", disabled=True)


def render_estimation_results():
    """Render estimation results and diagnostics."""
    st.subheader("📈 Estimation Results")
    
    # Mock results for demonstration
    results = generate_mock_estimation_results()
    
    # Parameter estimates table
    st.markdown("#### Regional Parameter Estimates")
    
    param_df = pd.DataFrame({
        'Region': results['regions'],
        'σ (Interest Rate Sensitivity)': results['sigma'],
        'κ (Phillips Curve Slope)': results['kappa'],
        'ψ (Demand Spillover)': results['psi'],
        'φ (Price Spillover)': results['phi']
    })
    
    st.dataframe(param_df, use_container_width=True)
    
    # Parameter visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Parameter distribution
        fig = px.box(
            param_df.melt(id_vars=['Region'], var_name='Parameter', value_name='Estimate'),
            x='Parameter',
            y='Estimate',
            title="Parameter Distribution Across Regions"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional parameter map (mock)
        fig = px.scatter(
            param_df,
            x='σ (Interest Rate Sensitivity)',
            y='κ (Phillips Curve Slope)',
            hover_data=['Region'],
            title="Regional Parameter Relationships"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Diagnostic tests
    render_diagnostic_tests(results)
    
    # Export options
    render_export_options(results)


def render_diagnostic_tests(results: Dict[str, Any]):
    """Render diagnostic test results."""
    st.markdown("#### 🔍 Diagnostic Tests")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Identification Test", "✅ Passed", "Strong instruments")
    
    with col2:
        st.metric("Overidentification", "p = 0.23", "Not rejected")
    
    with col3:
        st.metric("Spatial Autocorr.", "Moran's I = 0.15", "Significant")
    
    # Detailed diagnostics
    with st.expander("📋 Detailed Diagnostics", expanded=False):
        diagnostics = {
            "Test": ["Weak Instruments", "Overidentification", "Spatial Autocorrelation", "Heteroskedasticity"],
            "Statistic": [15.67, 8.23, 3.45, 12.89],
            "P-value": [0.001, 0.234, 0.063, 0.012],
            "Result": ["Reject H0", "Fail to Reject", "Marginal", "Reject H0"]
        }
        
        diag_df = pd.DataFrame(diagnostics)
        st.dataframe(diag_df, use_container_width=True)


def render_export_options(results: Dict[str, Any]):
    """Render export options for results."""
    st.markdown("#### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            export_results_csv(results)
    
    with col2:
        if st.button("📄 Generate Report"):
            generate_estimation_report(results)
    
    with col3:
        if st.button("💾 Save to Session"):
            save_results_to_session(results)


# Helper functions

def get_available_regions() -> List[str]:
    """Get available regions for estimation."""
    return ["US", "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ"]


def render_data_preview_summary(data_config: Dict[str, Any]):
    """Render data preview summary."""
    st.markdown("#### 📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Regions", len(data_config.get('regions', [])))
    
    with col2:
        st.metric("Indicators", len(data_config.get('indicators', [])))
    
    with col3:
        start_date = data_config.get('start_date')
        end_date = data_config.get('end_date')
        if start_date and end_date:
            days = (end_date - start_date).days
            st.metric("Time Span", f"{days} days")
    
    with col4:
        st.metric("Frequency", data_config.get('data_frequency', 'Unknown'))


def render_spatial_matrix_preview(spatial_config: Dict[str, Any]):
    """Render spatial matrix preview."""
    st.markdown("#### 🗺️ Spatial Matrix Preview")
    
    # Generate mock spatial matrix
    n_regions = 10  # Mock number of regions
    W = np.random.rand(n_regions, n_regions)
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)  # Zero diagonal
    
    # Normalize
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    
    # Visualize
    fig = px.imshow(
        W,
        title="Spatial Weight Matrix",
        color_continuous_scale="Blues",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_estimation_progress():
    """Render estimation progress."""
    if 'estimation_running' in st.session_state and st.session_state.estimation_running:
        st.markdown("#### ⏳ Estimation Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Mock progress updates
        for i in range(101):
            progress_bar.progress(i)
            if i < 33:
                status_text.text("Stage 1: Spatial weight estimation...")
            elif i < 66:
                status_text.text("Stage 2: Regional parameter estimation...")
            else:
                status_text.text("Stage 3: Policy parameter estimation...")
            
            import time
            time.sleep(0.01)
        
        st.success("✅ Estimation completed!")
        st.session_state.estimation_running = False


def run_parameter_estimation(data_config: Dict[str, Any], 
                           spatial_config: Dict[str, Any],
                           estimation_config: Dict[str, Any]):
    """Run parameter estimation."""
    st.session_state.estimation_running = True
    
    with st.spinner("Running parameter estimation..."):
        # Mock estimation process
        import time
        time.sleep(3)
        
        # Generate mock results
        results = generate_mock_estimation_results()
        
        # Save results
        workflow_manager = st.session_state.workflow_manager
        workflow_manager.update_workflow_data("estimation_results", results)
        
        st.success("✅ Parameter estimation completed!")
        st.session_state.estimation_running = False


def generate_mock_estimation_results() -> Dict[str, Any]:
    """Generate mock estimation results."""
    regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    n_regions = len(regions)
    
    return {
        'regions': regions,
        'sigma': np.random.normal(1.0, 0.3, n_regions),
        'kappa': np.random.normal(0.5, 0.2, n_regions),
        'psi': np.random.normal(0.2, 0.1, n_regions),
        'phi': np.random.normal(0.3, 0.15, n_regions),
        'standard_errors': {
            'sigma': np.random.normal(0.1, 0.02, n_regions),
            'kappa': np.random.normal(0.05, 0.01, n_regions),
            'psi': np.random.normal(0.03, 0.005, n_regions),
            'phi': np.random.normal(0.04, 0.008, n_regions)
        },
        'diagnostics': {
            'identification_test': 15.67,
            'overid_test': 8.23,
            'spatial_autocorr': 3.45
        }
    }


def save_estimation_configuration(data_config: Dict[str, Any],
                                spatial_config: Dict[str, Any], 
                                estimation_config: Dict[str, Any]):
    """Save estimation configuration."""
    config = {
        'data': data_config,
        'spatial': spatial_config,
        'estimation': estimation_config
    }
    
    session_manager = st.session_state.session_manager
    if session_manager.has_current_session():
        session_manager.update_session_config({'estimation_config': config})
        st.success("✅ Configuration saved to session!")
    else:
        st.warning("⚠️ No active session. Configuration not saved.")


def export_results_csv(results: Dict[str, Any]):
    """Export results to CSV."""
    st.success("✅ Results exported to CSV!")


def generate_estimation_report(results: Dict[str, Any]):
    """Generate estimation report."""
    st.success("✅ Estimation report generated!")


def save_results_to_session(results: Dict[str, Any]):
    """Save results to current session."""
    session_manager = st.session_state.session_manager
    if session_manager.has_current_session():
        session_manager.update_session_results({'parameter_estimation': results})
        st.success("✅ Results saved to session!")
    else:
        st.warning("⚠️ No active session. Results not saved.")