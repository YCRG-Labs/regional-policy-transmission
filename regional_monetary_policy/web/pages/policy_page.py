"""
Policy analysis page for monetary policy effectiveness and mistake decomposition.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go

from ..workflow_manager import WorkflowType


def render():
    """Render the policy analysis page."""
    st.title("📈 Policy Analysis")
    st.markdown("Analyze monetary policy effectiveness and decompose policy mistakes")
    st.markdown("---")
    
    # Check if we're in a workflow
    workflow_manager = st.session_state.workflow_manager
    current_workflow = workflow_manager.get_current_workflow()
    
    if current_workflow and current_workflow.workflow_type == WorkflowType.POLICY_ANALYSIS:
        render_workflow_policy_analysis()
    else:
        render_standalone_policy_analysis()


def render_workflow_policy_analysis():
    """Render policy analysis within guided workflow."""
    workflow_manager = st.session_state.workflow_manager
    workflow = workflow_manager.get_current_workflow()
    current_step = workflow.get_current_step()
    
    if not current_step:
        st.success("🎉 Policy analysis workflow completed!")
        render_policy_results()
        return
    
    st.info(f"**Current Step**: {current_step.title}")
    st.write(current_step.description)
    
    # Render appropriate step
    if current_step.step_id == "parameter_input":
        render_parameter_input_step()
    elif current_step.step_id == "policy_period":
        render_policy_period_step()
    elif current_step.step_id == "fed_weights":
        render_fed_weights_step()
    elif current_step.step_id == "mistake_decomposition":
        render_mistake_decomposition_step()
    elif current_step.step_id == "policy_results":
        render_policy_results_step()
    
    # Navigation buttons
    render_workflow_navigation()


def render_standalone_policy_analysis():
    """Render standalone policy analysis interface."""
    st.info("💡 For guided analysis, start a Policy Analysis workflow from the Home page")
    
    tabs = st.tabs(["Parameters", "Policy Period", "Fed Weights", "Decomposition", "Results"])
    
    with tabs[0]:
        render_parameter_input_step()
    
    with tabs[1]:
        render_policy_period_step()
    
    with tabs[2]:
        render_fed_weights_step()
    
    with tabs[3]:
        render_mistake_decomposition_step()
    
    with tabs[4]:
        render_policy_results()


def render_parameter_input_step():
    """Render parameter input step."""
    st.subheader("🔢 Parameter Input")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("parameter_input")
    
    # Parameter source selection
    parameter_source = st.radio(
        "Parameter Source",
        ["Load from Session", "Load from File", "Manual Input"],
        index=0,
        help="Choose how to provide regional parameters"
    )
    
    if parameter_source == "Load from Session":
        render_session_parameter_loader()
    elif parameter_source == "Load from File":
        render_file_parameter_loader()
    else:
        render_manual_parameter_input()


def render_session_parameter_loader():
    """Render session parameter loader."""
    session_manager = st.session_state.session_manager
    
    if not session_manager.has_current_session():
        st.warning("⚠️ No active session. Please create or load a session first.")
        return
    
    current_session = session_manager.get_current_session()
    results = current_session.results
    
    # Check for parameter estimation results
    if 'parameter_estimation' in results:
        st.success("✅ Parameter estimation results found in session")
        
        params = results['parameter_estimation']
        render_parameter_summary(params)
        
        if st.button("📊 Use These Parameters", type="primary"):
            workflow_manager = st.session_state.workflow_manager
            workflow_manager.update_workflow_data("parameter_input", {
                'source': 'session',
                'parameters': params
            })
            st.success("Parameters loaded successfully!")
    else:
        st.info("No parameter estimation results found in current session")
        st.markdown("**Available results:**")
        for key in results.keys():
            st.write(f"• {key}")


def render_file_parameter_loader():
    """Render file parameter loader."""
    st.markdown("#### 📁 Load Parameters from File")
    
    uploaded_file = st.file_uploader(
        "Choose parameter file",
        type=['csv', 'json', 'xlsx'],
        help="Upload file containing regional parameter estimates"
    )
    
    if uploaded_file is not None:
        try:
            # Load file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                import json
                data = json.load(uploaded_file)
                df = pd.DataFrame(data)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ File loaded successfully!")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate parameter structure
            required_columns = ['region', 'sigma', 'kappa', 'psi', 'phi']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
            else:
                if st.button("📊 Use These Parameters", type="primary"):
                    params = convert_dataframe_to_parameters(df)
                    workflow_manager = st.session_state.workflow_manager
                    workflow_manager.update_workflow_data("parameter_input", {
                        'source': 'file',
                        'parameters': params
                    })
                    st.success("Parameters loaded successfully!")
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")


def render_manual_parameter_input():
    """Render manual parameter input."""
    st.markdown("#### ✏️ Manual Parameter Input")
    
    # Region selection
    available_regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    selected_regions = st.multiselect(
        "Select Regions",
        available_regions,
        default=available_regions[:5]
    )
    
    if selected_regions:
        st.markdown("#### Parameter Values")
        
        # Create input form for each region
        params_data = {}
        
        for region in selected_regions:
            st.markdown(f"**{region}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sigma = st.number_input(f"σ ({region})", value=1.0, step=0.1, key=f"sigma_{region}")
            with col2:
                kappa = st.number_input(f"κ ({region})", value=0.5, step=0.1, key=f"kappa_{region}")
            with col3:
                psi = st.number_input(f"ψ ({region})", value=0.2, step=0.1, key=f"psi_{region}")
            with col4:
                phi = st.number_input(f"φ ({region})", value=0.3, step=0.1, key=f"phi_{region}")
            
            params_data[region] = {
                'sigma': sigma,
                'kappa': kappa,
                'psi': psi,
                'phi': phi
            }
        
        if st.button("💾 Save Manual Parameters", type="primary"):
            workflow_manager = st.session_state.workflow_manager
            workflow_manager.update_workflow_data("parameter_input", {
                'source': 'manual',
                'parameters': params_data
            })
            st.success("Manual parameters saved successfully!")


def render_policy_period_step():
    """Render policy period selection step."""
    st.subheader("📅 Policy Period Selection")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("policy_period")
    
    with st.form("policy_period_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_start = st.date_input(
                "Analysis Start Date",
                value=step_data.get('analysis_start', pd.Timestamp('2008-01-01').date()),
                help="Start date for policy analysis"
            )
            
            analysis_end = st.date_input(
                "Analysis End Date",
                value=step_data.get('analysis_end', pd.Timestamp('2020-12-31').date()),
                help="End date for policy analysis"
            )
        
        with col2:
            # Specific episodes
            crisis_episodes = st.multiselect(
                "Focus on Specific Episodes",
                ["Financial Crisis (2008-2009)", "Zero Lower Bound (2009-2015)", 
                 "Taper Tantrum (2013)", "COVID-19 (2020-2021)"],
                default=step_data.get('specific_episodes', [])
            )
            
            analysis_frequency = st.selectbox(
                "Analysis Frequency",
                ["Monthly", "Quarterly"],
                index=0 if step_data.get('analysis_frequency') == 'Monthly' else 1
            )
        
        if st.form_submit_button("📅 Set Policy Period", type="primary"):
            period_config = {
                'analysis_start': analysis_start,
                'analysis_end': analysis_end,
                'specific_episodes': crisis_episodes,
                'analysis_frequency': analysis_frequency
            }
            
            workflow_manager.update_workflow_data("policy_period", period_config)
            st.success("✅ Policy period configured!")
    
    # Display period summary
    if step_data:
        render_period_summary(step_data)


def render_fed_weights_step():
    """Render Fed weights estimation step."""
    st.subheader("🏛️ Fed Weight Estimation")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("fed_weights")
    
    st.markdown("#### Reaction Function Specification")
    
    with st.form("fed_weights_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            reaction_function_spec = st.selectbox(
                "Reaction Function Type",
                ["Taylor Rule", "Augmented Taylor Rule", "Forecast-Based Rule"],
                index=0,
                help="Specification for Fed reaction function"
            )
            
            include_regional_vars = st.checkbox(
                "Include Regional Variables",
                value=step_data.get('include_regional_vars', True),
                help="Include regional economic conditions in reaction function"
            )
            
            lag_structure = st.selectbox(
                "Lag Structure",
                ["No Lags", "1 Lag", "2 Lags", "4 Lags"],
                index=1
            )
        
        with col2:
            instrument_variables = st.multiselect(
                "Instrument Variables",
                ["Lagged Interest Rates", "Commodity Prices", "External Shocks", 
                 "Financial Conditions", "Survey Expectations"],
                default=step_data.get('instrument_variables', ["Lagged Interest Rates", "External Shocks"])
            )
            
            estimation_method = st.selectbox(
                "Estimation Method",
                ["OLS", "2SLS", "GMM"],
                index=1
            )
        
        if st.form_submit_button("🔧 Estimate Fed Weights", type="primary"):
            fed_config = {
                'reaction_function_spec': reaction_function_spec,
                'include_regional_vars': include_regional_vars,
                'lag_structure': lag_structure,
                'instrument_variables': instrument_variables,
                'estimation_method': estimation_method
            }
            
            # Run Fed weights estimation
            run_fed_weights_estimation(fed_config)
            
            workflow_manager.update_workflow_data("fed_weights", fed_config)
            st.success("✅ Fed weights estimated!")


def render_mistake_decomposition_step():
    """Render policy mistake decomposition step."""
    st.subheader("🔍 Policy Mistake Decomposition")
    
    workflow_manager = st.session_state.workflow_manager
    
    # Check prerequisites
    param_data = workflow_manager.get_workflow_data("parameter_input")
    period_data = workflow_manager.get_workflow_data("policy_period")
    fed_data = workflow_manager.get_workflow_data("fed_weights")
    
    if not all([param_data, period_data, fed_data]):
        st.warning("⚠️ Please complete all previous steps first")
        return
    
    st.markdown("#### Decomposition Configuration")
    
    with st.form("decomposition_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            decomposition_method = st.selectbox(
                "Decomposition Method",
                ["Theorem 4 (Full)", "Information Only", "Weights Only", "Parameters Only"],
                index=0,
                help="Choose decomposition approach"
            )
            
            welfare_weights_type = st.selectbox(
                "Welfare Weights",
                ["Population-Based", "GDP-Based", "Equal Weights", "Custom"],
                index=0
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.2f"
            )
            
            bootstrap_replications = st.number_input(
                "Bootstrap Replications",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        if st.form_submit_button("🚀 Run Decomposition", type="primary"):
            decomp_config = {
                'decomposition_method': decomposition_method,
                'welfare_weights_type': welfare_weights_type,
                'confidence_level': confidence_level,
                'bootstrap_replications': bootstrap_replications
            }
            
            # Run mistake decomposition
            run_mistake_decomposition(param_data, period_data, fed_data, decomp_config)
            
            workflow_manager.update_workflow_data("mistake_decomposition", decomp_config)


def render_policy_results_step():
    """Render policy analysis results step."""
    st.subheader("📊 Policy Analysis Results")
    
    workflow_manager = st.session_state.workflow_manager
    results = workflow_manager.get_workflow_data("decomposition_results")
    
    if not results:
        st.info("No decomposition results available. Please run mistake decomposition first.")
        return
    
    render_policy_results()


def render_policy_results():
    """Render comprehensive policy analysis results."""
    st.subheader("📈 Policy Analysis Results")
    
    # Generate mock results
    results = generate_mock_policy_results()
    
    # Mistake decomposition summary
    render_mistake_decomposition_summary(results)
    
    # Time series of policy mistakes
    render_policy_mistake_timeseries(results)
    
    # Regional analysis
    render_regional_policy_analysis(results)
    
    # Fed weights analysis
    render_fed_weights_analysis(results)


def render_mistake_decomposition_summary(results: Dict[str, Any]):
    """Render mistake decomposition summary."""
    st.markdown("#### 🎯 Policy Mistake Decomposition")
    
    decomp = results['decomposition']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Mistake",
            f"{decomp['total']:.3f}",
            help="Total policy mistake (percentage points)"
        )
    
    with col2:
        st.metric(
            "Information Effect",
            f"{decomp['information']:.3f}",
            f"{decomp['information']/decomp['total']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Weight Effect",
            f"{decomp['weights']:.3f}",
            f"{decomp['weights']/decomp['total']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "Parameter Effect",
            f"{decomp['parameters']:.3f}",
            f"{decomp['parameters']/decomp['total']*100:.1f}%"
        )
    
    # Decomposition chart
    fig = px.bar(
        x=['Information', 'Weights', 'Parameters', 'Inflation Response'],
        y=[decomp['information'], decomp['weights'], decomp['parameters'], decomp['inflation']],
        title="Policy Mistake Decomposition",
        labels={'x': 'Component', 'y': 'Contribution (pp)'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_policy_mistake_timeseries(results: Dict[str, Any]):
    """Render time series of policy mistakes."""
    st.markdown("#### 📈 Policy Mistakes Over Time")
    
    # Generate time series data
    dates = pd.date_range('2008-01-01', '2020-12-31', freq='M')
    n_periods = len(dates)
    
    mistake_data = pd.DataFrame({
        'Date': dates,
        'Total Mistake': np.random.normal(0, 0.5, n_periods),
        'Information Effect': np.random.normal(0, 0.2, n_periods),
        'Weight Effect': np.random.normal(0, 0.15, n_periods),
        'Parameter Effect': np.random.normal(0, 0.1, n_periods)
    })
    
    # Interactive time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=mistake_data['Date'],
        y=mistake_data['Total Mistake'],
        name='Total Mistake',
        line=dict(width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=mistake_data['Date'],
        y=mistake_data['Information Effect'],
        name='Information Effect',
        line=dict(dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=mistake_data['Date'],
        y=mistake_data['Weight Effect'],
        name='Weight Effect',
        line=dict(dash='dot')
    ))
    
    fig.update_layout(
        title="Policy Mistakes Over Time",
        xaxis_title="Date",
        yaxis_title="Policy Mistake (pp)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_regional_policy_analysis(results: Dict[str, Any]):
    """Render regional policy analysis."""
    st.markdown("#### 🗺️ Regional Policy Effects")
    
    # Regional impact data
    regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    regional_data = pd.DataFrame({
        'Region': regions,
        'Policy Impact': np.random.normal(0, 0.3, len(regions)),
        'Welfare Loss': np.random.uniform(0, 2, len(regions)),
        'Optimal Weight': np.random.uniform(0.05, 0.15, len(regions)),
        'Fed Weight': np.random.uniform(0.05, 0.15, len(regions))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional policy impact
        fig = px.bar(
            regional_data,
            x='Region',
            y='Policy Impact',
            title="Regional Policy Impact",
            color='Policy Impact',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weight comparison
        fig = px.scatter(
            regional_data,
            x='Fed Weight',
            y='Optimal Weight',
            hover_data=['Region'],
            title="Fed vs Optimal Weights"
        )
        # Add 45-degree line
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0.15, y1=0.15,
            line=dict(dash="dash", color="gray")
        )
        st.plotly_chart(fig, use_container_width=True)


def render_fed_weights_analysis(results: Dict[str, Any]):
    """Render Fed weights analysis."""
    st.markdown("#### 🏛️ Fed Regional Weights Analysis")
    
    # Fed weights over time
    dates = pd.date_range('2008-01-01', '2020-12-31', freq='Q')
    regions = ["CA", "NY", "TX", "FL", "IL"]
    
    weights_data = {}
    for region in regions:
        weights_data[region] = np.random.uniform(0.15, 0.25, len(dates))
    
    weights_df = pd.DataFrame(weights_data, index=dates)
    
    fig = go.Figure()
    for region in regions:
        fig.add_trace(go.Scatter(
            x=weights_df.index,
            y=weights_df[region],
            name=region,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Fed Regional Weights Over Time",
        xaxis_title="Date",
        yaxis_title="Implicit Weight",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


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


# Helper functions

def render_parameter_summary(params: Dict[str, Any]):
    """Render parameter summary."""
    st.markdown("#### 📊 Parameter Summary")
    
    if 'regions' in params:
        regions = params['regions']
        param_df = pd.DataFrame({
            'Region': regions,
            'σ': params.get('sigma', [0] * len(regions)),
            'κ': params.get('kappa', [0] * len(regions)),
            'ψ': params.get('psi', [0] * len(regions)),
            'φ': params.get('phi', [0] * len(regions))
        })
        
        st.dataframe(param_df, use_container_width=True)


def convert_dataframe_to_parameters(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert dataframe to parameters dictionary."""
    return {
        'regions': df['region'].tolist(),
        'sigma': df['sigma'].tolist(),
        'kappa': df['kappa'].tolist(),
        'psi': df['psi'].tolist(),
        'phi': df['phi'].tolist()
    }


def render_period_summary(period_data: Dict[str, Any]):
    """Render policy period summary."""
    st.markdown("#### 📅 Period Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = period_data.get('analysis_start')
        if start_date:
            st.write(f"**Start**: {start_date}")
    
    with col2:
        end_date = period_data.get('analysis_end')
        if end_date:
            st.write(f"**End**: {end_date}")
    
    with col3:
        frequency = period_data.get('analysis_frequency', 'Unknown')
        st.write(f"**Frequency**: {frequency}")
    
    episodes = period_data.get('specific_episodes', [])
    if episodes:
        st.write(f"**Episodes**: {', '.join(episodes)}")


def run_fed_weights_estimation(config: Dict[str, Any]):
    """Run Fed weights estimation."""
    with st.spinner("Estimating Fed regional weights..."):
        import time
        time.sleep(2)
        
        # Generate mock Fed weights
        regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        fed_weights = {
            'regions': regions,
            'weights': np.random.uniform(0.05, 0.15, len(regions)),
            'standard_errors': np.random.uniform(0.01, 0.03, len(regions))
        }
        
        workflow_manager = st.session_state.workflow_manager
        workflow_manager.update_workflow_data("fed_weights_results", fed_weights)
        
        st.success("✅ Fed weights estimated successfully!")


def run_mistake_decomposition(param_data: Dict[str, Any], period_data: Dict[str, Any],
                             fed_data: Dict[str, Any], decomp_config: Dict[str, Any]):
    """Run policy mistake decomposition."""
    with st.spinner("Running policy mistake decomposition..."):
        import time
        time.sleep(3)
        
        # Generate mock decomposition results
        results = generate_mock_policy_results()
        
        workflow_manager = st.session_state.workflow_manager
        workflow_manager.update_workflow_data("decomposition_results", results)
        
        st.success("✅ Policy mistake decomposition completed!")


def generate_mock_policy_results() -> Dict[str, Any]:
    """Generate mock policy analysis results."""
    return {
        'decomposition': {
            'total': 0.45,
            'information': 0.20,
            'weights': 0.15,
            'parameters': 0.08,
            'inflation': 0.02
        },
        'fed_weights': {
            'CA': 0.12,
            'NY': 0.18,
            'TX': 0.11,
            'FL': 0.08,
            'IL': 0.07
        },
        'optimal_weights': {
            'CA': 0.15,
            'NY': 0.14,
            'TX': 0.13,
            'FL': 0.10,
            'IL': 0.09
        },
        'welfare_loss': 2.34,
        'period': {
            'start': '2008-01-01',
            'end': '2020-12-31'
        }
    }