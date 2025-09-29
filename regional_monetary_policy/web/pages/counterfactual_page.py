"""
Counterfactual analysis page for evaluating alternative policy scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go

from ..workflow_manager import WorkflowType


def render():
    """Render the counterfactual analysis page."""
    st.title("🔄 Counterfactual Analysis")
    st.markdown("Evaluate welfare implications of alternative monetary policy approaches")
    st.markdown("---")
    
    # Check if we're in a workflow
    workflow_manager = st.session_state.workflow_manager
    current_workflow = workflow_manager.get_current_workflow()
    
    if current_workflow and current_workflow.workflow_type == WorkflowType.COUNTERFACTUAL_ANALYSIS:
        render_workflow_counterfactual()
    else:
        render_standalone_counterfactual()


def render_workflow_counterfactual():
    """Render counterfactual analysis within guided workflow."""
    workflow_manager = st.session_state.workflow_manager
    workflow = workflow_manager.get_current_workflow()
    current_step = workflow.get_current_step()
    
    if not current_step:
        st.success("🎉 Counterfactual analysis workflow completed!")
        render_counterfactual_results()
        return
    
    st.info(f"**Current Step**: {current_step.title}")
    st.write(current_step.description)
    
    # Render appropriate step
    if current_step.step_id == "baseline_setup":
        render_baseline_setup_step()
    elif current_step.step_id == "alternative_scenarios":
        render_alternative_scenarios_step()
    elif current_step.step_id == "welfare_function":
        render_welfare_function_step()
    elif current_step.step_id == "counterfactual_computation":
        render_counterfactual_computation_step()
    elif current_step.step_id == "welfare_comparison":
        render_welfare_comparison_step()
    
    # Navigation buttons
    render_workflow_navigation()


def render_standalone_counterfactual():
    """Render standalone counterfactual analysis interface."""
    st.info("💡 For guided analysis, start a Counterfactual Analysis workflow from the Home page")
    
    tabs = st.tabs(["Baseline", "Scenarios", "Welfare Function", "Computation", "Results"])
    
    with tabs[0]:
        render_baseline_setup_step()
    
    with tabs[1]:
        render_alternative_scenarios_step()
    
    with tabs[2]:
        render_welfare_function_step()
    
    with tabs[3]:
        render_counterfactual_computation_step()
    
    with tabs[4]:
        render_counterfactual_results()


def render_baseline_setup_step():
    """Render baseline scenario setup step."""
    st.subheader("📊 Baseline Scenario Setup")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("baseline_setup")
    
    with st.form("baseline_setup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_period_start = st.date_input(
                "Baseline Period Start",
                value=step_data.get('baseline_period_start', pd.Timestamp('2008-01-01').date()),
                help="Start date for baseline policy scenario"
            )
            
            baseline_period_end = st.date_input(
                "Baseline Period End",
                value=step_data.get('baseline_period_end', pd.Timestamp('2020-12-31').date()),
                help="End date for baseline policy scenario"
            )
            
            baseline_policy_type = st.selectbox(
                "Baseline Policy Type",
                ["Historical Fed Policy", "Taylor Rule", "Inflation Targeting", "Custom Rule"],
                index=0,
                help="Type of baseline monetary policy"
            )
        
        with col2:
            policy_parameters = {}
            
            if baseline_policy_type == "Taylor Rule":
                policy_parameters['inflation_target'] = st.number_input(
                    "Inflation Target (%)", value=2.0, step=0.1
                )
                policy_parameters['inflation_coefficient'] = st.number_input(
                    "Inflation Coefficient", value=1.5, step=0.1
                )
                policy_parameters['output_coefficient'] = st.number_input(
                    "Output Gap Coefficient", value=0.5, step=0.1
                )
            
            elif baseline_policy_type == "Custom Rule":
                st.markdown("**Custom Rule Parameters**")
                policy_parameters['custom_rule'] = st.text_area(
                    "Rule Specification",
                    value="r_t = 0.5 * r_{t-1} + 0.5 * (2.0 + 1.5 * π_t + 0.5 * y_t)",
                    help="Specify custom policy rule"
                )
            
            baseline_assumptions = st.multiselect(
                "Baseline Assumptions",
                ["Perfect Information", "Real-time Data", "Measurement Error", "Model Uncertainty"],
                default=step_data.get('baseline_assumptions', ["Real-time Data"])
            )
        
        if st.form_submit_button("📊 Configure Baseline", type="primary"):
            baseline_config = {
                'baseline_period_start': baseline_period_start,
                'baseline_period_end': baseline_period_end,
                'baseline_policy_type': baseline_policy_type,
                'policy_parameters': policy_parameters,
                'baseline_assumptions': baseline_assumptions
            }
            
            workflow_manager.update_workflow_data("baseline_setup", baseline_config)
            st.success("✅ Baseline scenario configured!")
    
    # Display baseline summary
    if step_data:
        render_baseline_summary(step_data)


def render_alternative_scenarios_step():
    """Render alternative scenarios step."""
    st.subheader("🔄 Alternative Scenarios")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("alternative_scenarios")
    
    st.markdown("#### Standard Scenarios")
    
    # Standard scenario selection
    standard_scenarios = st.multiselect(
        "Select Standard Scenarios",
        [
            "Perfect Information (PI): Fed has perfect real-time information",
            "Optimal Regional (OR): Fed uses optimal regional weights", 
            "Perfect Regional (PR): Perfect information + optimal weights"
        ],
        default=step_data.get('standard_scenarios', [
            "Perfect Information (PI): Fed has perfect real-time information",
            "Optimal Regional (OR): Fed uses optimal regional weights"
        ])
    )
    
    # Custom scenarios
    st.markdown("#### Custom Scenarios")
    
    custom_scenarios = []
    num_custom = st.number_input("Number of Custom Scenarios", min_value=0, max_value=5, value=0)
    
    for i in range(num_custom):
        with st.expander(f"Custom Scenario {i+1}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                scenario_name = st.text_input(f"Scenario Name", key=f"custom_name_{i}")
                scenario_description = st.text_area(f"Description", key=f"custom_desc_{i}")
            
            with col2:
                policy_rule = st.selectbox(
                    "Policy Rule",
                    ["Modified Taylor Rule", "Forecast-Based Rule", "Regional-Specific Rule"],
                    key=f"custom_rule_{i}"
                )
                
                regional_weights = st.selectbox(
                    "Regional Weights",
                    ["Population-Based", "GDP-Based", "Equal Weights", "Custom Weights"],
                    key=f"custom_weights_{i}"
                )
            
            if scenario_name:
                custom_scenarios.append({
                    'name': scenario_name,
                    'description': scenario_description,
                    'policy_rule': policy_rule,
                    'regional_weights': regional_weights
                })
    
    if st.button("💾 Save Scenarios", type="primary"):
        scenarios_config = {
            'standard_scenarios': standard_scenarios,
            'custom_scenarios': custom_scenarios
        }
        
        workflow_manager.update_workflow_data("alternative_scenarios", scenarios_config)
        st.success("✅ Alternative scenarios configured!")
    
    # Display scenarios summary
    if step_data or standard_scenarios or custom_scenarios:
        render_scenarios_summary(standard_scenarios, custom_scenarios)


def render_welfare_function_step():
    """Render welfare function setup step."""
    st.subheader("⚖️ Welfare Function Setup")
    
    workflow_manager = st.session_state.workflow_manager
    step_data = workflow_manager.get_workflow_data("welfare_function")
    
    with st.form("welfare_function_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Regional Weights")
            
            welfare_weights_type = st.selectbox(
                "Weight Type",
                ["Population-Based", "GDP-Based", "Equal Weights", "Custom Weights"],
                index=0,
                help="How to weight regional welfare in social welfare function"
            )
            
            if welfare_weights_type == "Custom Weights":
                st.markdown("**Custom Regional Weights**")
                regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
                custom_weights = {}
                
                for region in regions:
                    weight = st.number_input(
                        f"{region} Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0/len(regions),
                        step=0.01,
                        key=f"weight_{region}"
                    )
                    custom_weights[region] = weight
                
                # Normalize weights
                total_weight = sum(custom_weights.values())
                if total_weight > 0:
                    custom_weights = {k: v/total_weight for k, v in custom_weights.items()}
        
        with col2:
            st.markdown("#### Preference Parameters")
            
            discount_factor = st.slider(
                "Discount Factor (β)",
                min_value=0.90,
                max_value=0.99,
                value=step_data.get('discount_factor', 0.96),
                step=0.01,
                help="Intertemporal discount factor"
            )
            
            risk_aversion = st.slider(
                "Risk Aversion (γ)",
                min_value=1.0,
                max_value=10.0,
                value=step_data.get('risk_aversion', 2.0),
                step=0.1,
                help="Coefficient of relative risk aversion"
            )
            
            inflation_aversion = st.slider(
                "Inflation Aversion (λ)",
                min_value=0.0,
                max_value=2.0,
                value=step_data.get('inflation_aversion', 0.5),
                step=0.1,
                help="Relative weight on inflation stabilization"
            )
            
            welfare_function_type = st.selectbox(
                "Welfare Function Type",
                ["Quadratic Loss", "CRRA Utility", "Epstein-Zin"],
                index=0
            )
        
        if st.form_submit_button("⚖️ Configure Welfare Function", type="primary"):
            welfare_config = {
                'welfare_weights_type': welfare_weights_type,
                'discount_factor': discount_factor,
                'risk_aversion': risk_aversion,
                'inflation_aversion': inflation_aversion,
                'welfare_function_type': welfare_function_type
            }
            
            if welfare_weights_type == "Custom Weights":
                welfare_config['custom_weights'] = custom_weights
            
            workflow_manager.update_workflow_data("welfare_function", welfare_config)
            st.success("✅ Welfare function configured!")
    
    # Display welfare function preview
    if step_data:
        render_welfare_function_preview(step_data)


def render_counterfactual_computation_step():
    """Render counterfactual computation step."""
    st.subheader("🚀 Counterfactual Computation")
    
    workflow_manager = st.session_state.workflow_manager
    
    # Check prerequisites
    baseline_data = workflow_manager.get_workflow_data("baseline_setup")
    scenarios_data = workflow_manager.get_workflow_data("alternative_scenarios")
    welfare_data = workflow_manager.get_workflow_data("welfare_function")
    
    if not all([baseline_data, scenarios_data, welfare_data]):
        st.warning("⚠️ Please complete all previous configuration steps first")
        return
    
    # Display computation configuration
    with st.expander("📋 Computation Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Baseline Setup**")
            st.write(f"Period: {baseline_data.get('baseline_period_start')} to {baseline_data.get('baseline_period_end')}")
            st.write(f"Policy: {baseline_data.get('baseline_policy_type')}")
        
        with col2:
            st.markdown("**Scenarios**")
            standard_count = len(scenarios_data.get('standard_scenarios', []))
            custom_count = len(scenarios_data.get('custom_scenarios', []))
            st.write(f"Standard: {standard_count}")
            st.write(f"Custom: {custom_count}")
        
        with col3:
            st.markdown("**Welfare Function**")
            st.write(f"Weights: {welfare_data.get('welfare_weights_type')}")
            st.write(f"Risk Aversion: {welfare_data.get('risk_aversion')}")
    
    # Computation options
    st.markdown("#### Computation Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        parallel_processing = st.checkbox(
            "Enable Parallel Processing",
            value=True,
            help="Use multiple CPU cores for faster computation"
        )
        
        monte_carlo_runs = st.number_input(
            "Monte Carlo Runs",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of Monte Carlo simulations for uncertainty quantification"
        )
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
        
        save_intermediate = st.checkbox(
            "Save Intermediate Results",
            value=True,
            help="Save intermediate results for debugging"
        )
    
    # Run computation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run Counterfactual Analysis", type="primary", use_container_width=True):
            run_counterfactual_computation(
                baseline_data, scenarios_data, welfare_data,
                parallel_processing, monte_carlo_runs, confidence_level, save_intermediate
            )
    
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            save_counterfactual_configuration(baseline_data, scenarios_data, welfare_data)
    
    # Display computation progress
    render_computation_progress()


def render_welfare_comparison_step():
    """Render welfare comparison step."""
    st.subheader("📊 Welfare Comparison")
    
    workflow_manager = st.session_state.workflow_manager
    results = workflow_manager.get_workflow_data("counterfactual_results")
    
    if not results:
        st.info("No counterfactual results available. Please run computation first.")
        return
    
    render_counterfactual_results()


def render_counterfactual_results():
    """Render comprehensive counterfactual analysis results."""
    st.subheader("📈 Counterfactual Analysis Results")
    
    # Generate mock results
    results = generate_mock_counterfactual_results()
    
    # Welfare ranking verification
    render_welfare_ranking(results)
    
    # Scenario comparison
    render_scenario_comparison(results)
    
    # Regional welfare impacts
    render_regional_welfare_impacts(results)
    
    # Policy implications
    render_policy_implications(results)


def render_welfare_ranking(results: Dict[str, Any]):
    """Render welfare ranking verification."""
    st.markdown("#### 🏆 Welfare Ranking")
    
    welfare_outcomes = results['welfare_outcomes']
    
    # Verify theoretical ranking: W^PR ≥ W^PI ≥ W^OR ≥ W^B
    ranking_df = pd.DataFrame({
        'Scenario': ['Perfect Regional (PR)', 'Perfect Information (PI)', 'Optimal Regional (OR)', 'Baseline (B)'],
        'Welfare': [welfare_outcomes['perfect_regional'], welfare_outcomes['perfect_information'], 
                   welfare_outcomes['optimal_regional'], welfare_outcomes['baseline']],
        'Gain vs Baseline': [
            welfare_outcomes['perfect_regional'] - welfare_outcomes['baseline'],
            welfare_outcomes['perfect_information'] - welfare_outcomes['baseline'],
            welfare_outcomes['optimal_regional'] - welfare_outcomes['baseline'],
            0
        ]
    })
    
    # Sort by welfare (descending)
    ranking_df = ranking_df.sort_values('Welfare', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(ranking_df, use_container_width=True)
        
        # Check theoretical ranking
        theoretical_order = ['Perfect Regional (PR)', 'Perfect Information (PI)', 'Optimal Regional (OR)', 'Baseline (B)']
        actual_order = ranking_df['Scenario'].tolist()
        
        if actual_order == theoretical_order:
            st.success("✅ Welfare ranking matches theoretical prediction")
        else:
            st.warning("⚠️ Welfare ranking deviates from theoretical prediction")
    
    with col2:
        # Welfare gains chart
        fig = px.bar(
            ranking_df[ranking_df['Scenario'] != 'Baseline (B)'],
            x='Scenario',
            y='Gain vs Baseline',
            title="Welfare Gains Relative to Baseline",
            color='Gain vs Baseline',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_scenario_comparison(results: Dict[str, Any]):
    """Render detailed scenario comparison."""
    st.markdown("#### 🔄 Scenario Comparison")
    
    scenarios = results['scenario_details']
    
    # Time series comparison
    dates = pd.date_range('2008-01-01', '2020-12-31', freq='Q')
    
    fig = go.Figure()
    
    for scenario_name, scenario_data in scenarios.items():
        fig.add_trace(go.Scatter(
            x=dates,
            y=scenario_data['policy_rates'],
            name=scenario_name,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Policy Rates Across Scenarios",
        xaxis_title="Date",
        yaxis_title="Policy Rate (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")
    
    summary_data = []
    for scenario_name, scenario_data in scenarios.items():
        summary_data.append({
            'Scenario': scenario_name,
            'Avg Policy Rate': np.mean(scenario_data['policy_rates']),
            'Policy Volatility': np.std(scenario_data['policy_rates']),
            'Avg Inflation': np.mean(scenario_data['inflation']),
            'Output Gap Volatility': np.std(scenario_data['output_gaps'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)


def render_regional_welfare_impacts(results: Dict[str, Any]):
    """Render regional welfare impacts."""
    st.markdown("#### 🗺️ Regional Welfare Impacts")
    
    regional_impacts = results['regional_impacts']
    
    # Regional welfare gains
    regions = list(regional_impacts.keys())
    welfare_gains = [regional_impacts[region]['welfare_gain'] for region in regions]
    
    fig = px.bar(
        x=regions,
        y=welfare_gains,
        title="Regional Welfare Gains from Optimal Policy",
        labels={'x': 'Region', 'y': 'Welfare Gain'},
        color=welfare_gains,
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional impact details
    impact_df = pd.DataFrame({
        'Region': regions,
        'Welfare Gain': welfare_gains,
        'Inflation Reduction': [regional_impacts[region]['inflation_reduction'] for region in regions],
        'Output Stabilization': [regional_impacts[region]['output_stabilization'] for region in regions]
    })
    
    st.dataframe(impact_df, use_container_width=True)


def render_policy_implications(results: Dict[str, Any]):
    """Render policy implications."""
    st.markdown("#### 💡 Policy Implications")
    
    implications = results['policy_implications']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Findings**")
        for finding in implications['key_findings']:
            st.write(f"• {finding}")
    
    with col2:
        st.markdown("**Policy Recommendations**")
        for recommendation in implications['recommendations']:
            st.write(f"• {recommendation}")
    
    # Quantitative summary
    st.markdown("#### 📊 Quantitative Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Welfare Gain",
            f"{implications['max_welfare_gain']:.2f}%",
            "Perfect Regional Policy"
        )
    
    with col2:
        st.metric(
            "Information Value",
            f"{implications['information_value']:.2f}%",
            "Perfect vs Real-time Info"
        )
    
    with col3:
        st.metric(
            "Regional Weights Value",
            f"{implications['regional_weights_value']:.2f}%",
            "Optimal vs Equal Weights"
        )
    
    with col4:
        st.metric(
            "Combined Value",
            f"{implications['combined_value']:.2f}%",
            "Information + Weights"
        )


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

def render_baseline_summary(baseline_data: Dict[str, Any]):
    """Render baseline scenario summary."""
    st.markdown("#### 📊 Baseline Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Period**: {baseline_data.get('baseline_period_start')} to {baseline_data.get('baseline_period_end')}")
    
    with col2:
        st.write(f"**Policy Type**: {baseline_data.get('baseline_policy_type')}")
    
    with col3:
        assumptions = baseline_data.get('baseline_assumptions', [])
        st.write(f"**Assumptions**: {', '.join(assumptions)}")


def render_scenarios_summary(standard_scenarios: List[str], custom_scenarios: List[Dict[str, Any]]):
    """Render scenarios summary."""
    st.markdown("#### 🔄 Scenarios Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standard Scenarios**")
        for scenario in standard_scenarios:
            st.write(f"• {scenario.split(':')[0]}")
    
    with col2:
        st.markdown("**Custom Scenarios**")
        for scenario in custom_scenarios:
            st.write(f"• {scenario['name']}")


def render_welfare_function_preview(welfare_data: Dict[str, Any]):
    """Render welfare function preview."""
    st.markdown("#### ⚖️ Welfare Function Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Weights**: {welfare_data.get('welfare_weights_type')}")
    
    with col2:
        st.write(f"**Risk Aversion**: {welfare_data.get('risk_aversion')}")
    
    with col3:
        st.write(f"**Discount Factor**: {welfare_data.get('discount_factor')}")


def render_computation_progress():
    """Render computation progress."""
    if 'counterfactual_running' in st.session_state and st.session_state.counterfactual_running:
        st.markdown("#### ⏳ Computation Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Mock progress updates
        stages = [
            "Initializing scenarios...",
            "Computing baseline scenario...",
            "Running Perfect Information scenario...",
            "Running Optimal Regional scenario...",
            "Running Perfect Regional scenario...",
            "Computing welfare outcomes...",
            "Finalizing results..."
        ]
        
        for i, stage in enumerate(stages):
            progress = int((i + 1) / len(stages) * 100)
            progress_bar.progress(progress)
            status_text.text(stage)
            
            import time
            time.sleep(0.5)
        
        st.success("✅ Counterfactual computation completed!")
        st.session_state.counterfactual_running = False


def run_counterfactual_computation(baseline_data: Dict[str, Any], scenarios_data: Dict[str, Any],
                                 welfare_data: Dict[str, Any], parallel_processing: bool,
                                 monte_carlo_runs: int, confidence_level: float, save_intermediate: bool):
    """Run counterfactual computation."""
    st.session_state.counterfactual_running = True
    
    with st.spinner("Running counterfactual analysis..."):
        # Mock computation process
        import time
        time.sleep(5)
        
        # Generate mock results
        results = generate_mock_counterfactual_results()
        
        # Save results
        workflow_manager = st.session_state.workflow_manager
        workflow_manager.update_workflow_data("counterfactual_results", results)
        
        st.success("✅ Counterfactual analysis completed!")
        st.session_state.counterfactual_running = False


def save_counterfactual_configuration(baseline_data: Dict[str, Any], scenarios_data: Dict[str, Any], 
                                    welfare_data: Dict[str, Any]):
    """Save counterfactual configuration."""
    config = {
        'baseline': baseline_data,
        'scenarios': scenarios_data,
        'welfare': welfare_data
    }
    
    session_manager = st.session_state.session_manager
    if session_manager.has_current_session():
        session_manager.update_session_config({'counterfactual_config': config})
        st.success("✅ Configuration saved to session!")
    else:
        st.warning("⚠️ No active session. Configuration not saved.")


def generate_mock_counterfactual_results() -> Dict[str, Any]:
    """Generate mock counterfactual results."""
    n_periods = 52  # Quarterly data for 13 years
    
    return {
        'welfare_outcomes': {
            'baseline': -2.45,
            'perfect_information': -1.89,
            'optimal_regional': -2.12,
            'perfect_regional': -1.56
        },
        'scenario_details': {
            'Baseline': {
                'policy_rates': np.random.normal(2.5, 1.5, n_periods),
                'inflation': np.random.normal(2.0, 1.0, n_periods),
                'output_gaps': np.random.normal(0.0, 2.0, n_periods)
            },
            'Perfect Information': {
                'policy_rates': np.random.normal(2.3, 1.2, n_periods),
                'inflation': np.random.normal(2.0, 0.8, n_periods),
                'output_gaps': np.random.normal(0.0, 1.5, n_periods)
            },
            'Optimal Regional': {
                'policy_rates': np.random.normal(2.4, 1.3, n_periods),
                'inflation': np.random.normal(2.0, 0.9, n_periods),
                'output_gaps': np.random.normal(0.0, 1.7, n_periods)
            },
            'Perfect Regional': {
                'policy_rates': np.random.normal(2.2, 1.0, n_periods),
                'inflation': np.random.normal(2.0, 0.7, n_periods),
                'output_gaps': np.random.normal(0.0, 1.3, n_periods)
            }
        },
        'regional_impacts': {
            'CA': {'welfare_gain': 0.45, 'inflation_reduction': 0.12, 'output_stabilization': 0.23},
            'NY': {'welfare_gain': 0.38, 'inflation_reduction': 0.09, 'output_stabilization': 0.19},
            'TX': {'welfare_gain': 0.52, 'inflation_reduction': 0.15, 'output_stabilization': 0.28},
            'FL': {'welfare_gain': 0.41, 'inflation_reduction': 0.11, 'output_stabilization': 0.21},
            'IL': {'welfare_gain': 0.35, 'inflation_reduction': 0.08, 'output_stabilization': 0.17}
        },
        'policy_implications': {
            'key_findings': [
                "Perfect regional policy yields 36% welfare improvement over baseline",
                "Information improvements account for 60% of potential gains",
                "Regional weight optimization provides additional 25% improvement",
                "Combined reforms could reduce welfare losses by over one-third"
            ],
            'recommendations': [
                "Invest in real-time regional data collection and processing",
                "Develop regional economic indicators for policy decisions",
                "Consider regional heterogeneity in monetary policy framework",
                "Enhance communication about regional policy considerations"
            ],
            'max_welfare_gain': 36.3,
            'information_value': 22.9,
            'regional_weights_value': 13.5,
            'combined_value': 36.3
        }
    }