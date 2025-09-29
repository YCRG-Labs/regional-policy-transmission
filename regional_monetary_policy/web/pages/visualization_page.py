"""
Visualization page for interactive charts and regional maps.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    """Render the visualization page."""
    st.title("📋 Interactive Visualization")
    st.markdown("Create interactive charts and regional maps for analysis results")
    st.markdown("---")
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Regional Economic Maps",
            "Time Series Analysis", 
            "Parameter Estimation Results",
            "Policy Analysis Charts",
            "Counterfactual Comparisons",
            "Custom Visualizations"
        ]
    )
    
    if viz_type == "Regional Economic Maps":
        render_regional_maps()
    elif viz_type == "Time Series Analysis":
        render_time_series_analysis()
    elif viz_type == "Parameter Estimation Results":
        render_parameter_visualizations()
    elif viz_type == "Policy Analysis Charts":
        render_policy_visualizations()
    elif viz_type == "Counterfactual Comparisons":
        render_counterfactual_visualizations()
    elif viz_type == "Custom Visualizations":
        render_custom_visualizations()


def render_regional_maps():
    """Render regional economic maps."""
    st.subheader("🗺️ Regional Economic Maps")
    
    # Map configuration
    col1, col2 = st.columns(2)
    
    with col1:
        map_indicator = st.selectbox(
            "Economic Indicator",
            ["GDP Growth", "Inflation Rate", "Unemployment", "Output Gap", 
             "Parameter Estimates", "Policy Impact"]
        )
        
        time_period = st.selectbox(
            "Time Period",
            ["Latest", "2020", "2019", "2018", "2017", "2016", "Custom Range"]
        )
        
        if time_period == "Custom Range":
            date_range = st.date_input(
                "Select Date Range",
                value=(pd.Timestamp('2020-01-01').date(), pd.Timestamp('2020-12-31').date())
            )
    
    with col2:
        color_scale = st.selectbox(
            "Color Scale",
            ["Viridis", "RdYlBu", "RdBu", "Blues", "Reds", "Greens"]
        )
        
        map_projection = st.selectbox(
            "Map Projection",
            ["Natural Earth", "Albers USA", "Mercator", "Orthographic"]
        )
        
        show_state_borders = st.checkbox("Show State Borders", value=True)
    
    # Generate and display map
    if st.button("🗺️ Generate Map", type="primary"):
        render_choropleth_map(map_indicator, time_period, color_scale, map_projection, show_state_borders)
    
    # Additional map options
    with st.expander("🔧 Advanced Map Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            show_hover_data = st.checkbox("Show Hover Data", value=True)
            show_legend = st.checkbox("Show Legend", value=True)
            
        with col2:
            map_height = st.slider("Map Height", min_value=400, max_value=800, value=600)
            annotation_size = st.slider("Annotation Size", min_value=8, max_value=16, value=12)


def render_time_series_analysis():
    """Render time series analysis visualizations."""
    st.subheader("📈 Time Series Analysis")
    
    # Data selection
    col1, col2 = st.columns(2)
    
    with col1:
        regions = st.multiselect(
            "Select Regions",
            ["US", "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
            default=["US", "CA", "NY", "TX"]
        )
        
        indicators = st.multiselect(
            "Select Indicators", 
            ["GDP Growth", "Inflation", "Unemployment", "Interest Rate", "Output Gap"],
            default=["GDP Growth", "Inflation"]
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Area Chart", "Scatter Plot", "Box Plot", "Heatmap"]
        )
        
        time_aggregation = st.selectbox(
            "Time Aggregation",
            ["Monthly", "Quarterly", "Annual", "No Aggregation"]
        )
    
    # Generate time series data
    if regions and indicators:
        ts_data = generate_time_series_data(regions, indicators)
        
        if chart_type == "Line Chart":
            render_line_chart(ts_data, regions, indicators)
        elif chart_type == "Area Chart":
            render_area_chart(ts_data, regions, indicators)
        elif chart_type == "Scatter Plot":
            render_scatter_plot(ts_data, regions, indicators)
        elif chart_type == "Box Plot":
            render_box_plot(ts_data, regions, indicators)
        elif chart_type == "Heatmap":
            render_correlation_heatmap(ts_data, regions, indicators)
    
    # Time series analysis tools
    with st.expander("🔧 Analysis Tools", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Correlation Analysis"):
                render_correlation_analysis(ts_data if 'ts_data' in locals() else None)
            
            if st.button("📈 Trend Analysis"):
                render_trend_analysis(ts_data if 'ts_data' in locals() else None)
        
        with col2:
            if st.button("🔄 Seasonality Analysis"):
                render_seasonality_analysis(ts_data if 'ts_data' in locals() else None)
            
            if st.button("📉 Volatility Analysis"):
                render_volatility_analysis(ts_data if 'ts_data' in locals() else None)


def render_parameter_visualizations():
    """Render parameter estimation result visualizations."""
    st.subheader("🔢 Parameter Estimation Results")
    
    # Check for parameter results in session
    session_manager = st.session_state.session_manager
    
    if not session_manager.has_current_session():
        st.warning("⚠️ No active session. Please load a session with parameter estimation results.")
        return
    
    current_session = session_manager.get_current_session()
    param_results = current_session.results.get('parameter_estimation')
    
    if not param_results:
        st.info("No parameter estimation results found. Please run parameter estimation first.")
        
        # Generate mock results for demonstration
        if st.button("📊 Use Demo Data"):
            param_results = generate_mock_parameter_results()
    
    if param_results:
        # Parameter visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            viz_option = st.selectbox(
                "Visualization Type",
                ["Parameter Distribution", "Regional Comparison", "Confidence Intervals", 
                 "Parameter Relationships", "Diagnostic Plots"]
            )
        
        with col2:
            parameter_focus = st.selectbox(
                "Focus Parameter",
                ["All Parameters", "σ (Interest Rate Sensitivity)", "κ (Phillips Curve)", 
                 "ψ (Demand Spillover)", "φ (Price Spillover)"]
            )
        
        # Render selected visualization
        if viz_option == "Parameter Distribution":
            render_parameter_distribution(param_results, parameter_focus)
        elif viz_option == "Regional Comparison":
            render_regional_parameter_comparison(param_results, parameter_focus)
        elif viz_option == "Confidence Intervals":
            render_parameter_confidence_intervals(param_results, parameter_focus)
        elif viz_option == "Parameter Relationships":
            render_parameter_relationships(param_results)
        elif viz_option == "Diagnostic Plots":
            render_parameter_diagnostics(param_results)


def render_policy_visualizations():
    """Render policy analysis visualizations."""
    st.subheader("📈 Policy Analysis Charts")
    
    # Check for policy results
    session_manager = st.session_state.session_manager
    
    if session_manager.has_current_session():
        policy_results = session_manager.get_current_session().results.get('policy_analysis')
    else:
        policy_results = None
    
    if not policy_results:
        st.info("No policy analysis results found. Using demo data for visualization.")
        policy_results = generate_mock_policy_results()
    
    # Policy visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        policy_viz_type = st.selectbox(
            "Chart Type",
            ["Policy Mistake Decomposition", "Fed Weights Analysis", "Time Series of Mistakes",
             "Regional Policy Impact", "Welfare Loss Analysis"]
        )
    
    with col2:
        time_focus = st.selectbox(
            "Time Focus",
            ["Full Period", "Financial Crisis", "Zero Lower Bound", "Recent Years"]
        )
    
    # Render selected policy visualization
    if policy_viz_type == "Policy Mistake Decomposition":
        render_mistake_decomposition_chart(policy_results)
    elif policy_viz_type == "Fed Weights Analysis":
        render_fed_weights_chart(policy_results)
    elif policy_viz_type == "Time Series of Mistakes":
        render_policy_mistakes_timeseries(policy_results, time_focus)
    elif policy_viz_type == "Regional Policy Impact":
        render_regional_policy_impact(policy_results)
    elif policy_viz_type == "Welfare Loss Analysis":
        render_welfare_loss_analysis(policy_results)


def render_counterfactual_visualizations():
    """Render counterfactual analysis visualizations."""
    st.subheader("🔄 Counterfactual Comparisons")
    
    # Check for counterfactual results
    session_manager = st.session_state.session_manager
    
    if session_manager.has_current_session():
        cf_results = session_manager.get_current_session().results.get('counterfactual_analysis')
    else:
        cf_results = None
    
    if not cf_results:
        st.info("No counterfactual results found. Using demo data for visualization.")
        cf_results = generate_mock_counterfactual_results()
    
    # Counterfactual visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        cf_viz_type = st.selectbox(
            "Visualization Type",
            ["Welfare Ranking", "Scenario Comparison", "Policy Rate Paths", 
             "Regional Welfare Gains", "Decomposition Analysis"]
        )
    
    with col2:
        scenario_focus = st.multiselect(
            "Focus Scenarios",
            ["Baseline", "Perfect Information", "Optimal Regional", "Perfect Regional"],
            default=["Baseline", "Perfect Regional"]
        )
    
    # Render selected counterfactual visualization
    if cf_viz_type == "Welfare Ranking":
        render_welfare_ranking_chart(cf_results)
    elif cf_viz_type == "Scenario Comparison":
        render_scenario_comparison_chart(cf_results, scenario_focus)
    elif cf_viz_type == "Policy Rate Paths":
        render_policy_rate_paths(cf_results, scenario_focus)
    elif cf_viz_type == "Regional Welfare Gains":
        render_regional_welfare_gains(cf_results)
    elif cf_viz_type == "Decomposition Analysis":
        render_counterfactual_decomposition(cf_results)


def render_custom_visualizations():
    """Render custom visualization builder."""
    st.subheader("🎨 Custom Visualizations")
    
    st.markdown("Build custom visualizations using your analysis data.")
    
    # Data source selection
    data_source = st.selectbox(
        "Data Source",
        ["Session Results", "Upload File", "Manual Input", "Demo Data"]
    )
    
    if data_source == "Session Results":
        render_session_data_visualizer()
    elif data_source == "Upload File":
        render_file_upload_visualizer()
    elif data_source == "Manual Input":
        render_manual_input_visualizer()
    elif data_source == "Demo Data":
        render_demo_data_visualizer()


# Visualization rendering functions

def render_choropleth_map(indicator: str, time_period: str, color_scale: str, 
                         projection: str, show_borders: bool):
    """Render choropleth map."""
    # Generate mock regional data
    states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]
    values = np.random.normal(0, 1, len(states))
    
    # Create mock geographic data
    map_data = pd.DataFrame({
        'State': states,
        'Value': values,
        'State_Code': states  # In real implementation, would use proper state codes
    })
    
    # Create choropleth map (simplified version)
    fig = px.choropleth(
        map_data,
        locations='State_Code',
        color='Value',
        locationmode='USA-states',
        title=f"{indicator} - {time_period}",
        color_continuous_scale=color_scale,
        scope='usa'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=show_borders,
            projection_type=projection.lower().replace(' ', '')
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render line chart for time series data."""
    fig = make_subplots(
        rows=len(indicators), 
        cols=1,
        subplot_titles=indicators,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, indicator in enumerate(indicators):
        for j, region in enumerate(regions):
            col_name = f"{region}_{indicator}"
            if col_name in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col_name],
                        name=f"{region}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=300 * len(indicators),
        title="Regional Economic Indicators Over Time",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_parameter_distribution(param_results: Dict[str, Any], parameter_focus: str):
    """Render parameter distribution visualization."""
    regions = param_results['regions']
    
    if parameter_focus == "All Parameters":
        # Create subplot for all parameters
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['σ (Interest Rate Sensitivity)', 'κ (Phillips Curve)', 
                           'ψ (Demand Spillover)', 'φ (Price Spillover)']
        )
        
        params = ['sigma', 'kappa', 'psi', 'phi']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for param, (row, col) in zip(params, positions):
            values = param_results[param]
            fig.add_trace(
                go.Box(y=values, name=param, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title="Parameter Distribution Across Regions")
        
    else:
        # Focus on single parameter
        param_map = {
            'σ (Interest Rate Sensitivity)': 'sigma',
            'κ (Phillips Curve)': 'kappa',
            'ψ (Demand Spillover)': 'psi',
            'φ (Price Spillover)': 'phi'
        }
        
        param_key = param_map[parameter_focus]
        values = param_results[param_key]
        
        fig = go.Figure()
        
        # Add box plot
        fig.add_trace(go.Box(y=values, name=parameter_focus))
        
        # Add individual points
        fig.add_trace(go.Scatter(
            x=[parameter_focus] * len(values),
            y=values,
            mode='markers',
            name='Regional Values',
            text=regions,
            hovertemplate='%{text}: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{parameter_focus} Distribution",
            yaxis_title="Parameter Value"
        )
    
    st.plotly_chart(fig, use_container_width=True)


def render_mistake_decomposition_chart(policy_results: Dict[str, Any]):
    """Render policy mistake decomposition chart."""
    decomp = policy_results['decomposition']
    
    components = ['Information', 'Weights', 'Parameters', 'Inflation Response']
    values = [decomp['information'], decomp['weights'], decomp['parameters'], decomp['inflation']]
    
    # Stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Policy Mistake'],
        y=[values[0]],
        name=components[0],
        marker_color='lightblue'
    ))
    
    for i in range(1, len(components)):
        fig.add_trace(go.Bar(
            x=['Policy Mistake'],
            y=[values[i]],
            name=components[i],
            marker_color=px.colors.qualitative.Set1[i]
        ))
    
    fig.update_layout(
        barmode='stack',
        title="Policy Mistake Decomposition",
        yaxis_title="Contribution (percentage points)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_welfare_ranking_chart(cf_results: Dict[str, Any]):
    """Render welfare ranking chart."""
    welfare_outcomes = cf_results['welfare_outcomes']
    
    scenarios = list(welfare_outcomes.keys())
    welfare_values = list(welfare_outcomes.values())
    
    # Sort by welfare value (descending)
    sorted_data = sorted(zip(scenarios, welfare_values), key=lambda x: x[1], reverse=True)
    scenarios, welfare_values = zip(*sorted_data)
    
    fig = px.bar(
        x=scenarios,
        y=welfare_values,
        title="Welfare Ranking Across Scenarios",
        labels={'x': 'Scenario', 'y': 'Welfare Outcome'},
        color=welfare_values,
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Helper functions for generating mock data

def generate_time_series_data(regions: List[str], indicators: List[str]) -> pd.DataFrame:
    """Generate mock time series data."""
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='M')
    data = {}
    
    for region in regions:
        for indicator in indicators:
            col_name = f"{region}_{indicator}"
            
            # Generate realistic-looking data based on indicator type
            if indicator == "GDP Growth":
                data[col_name] = np.random.normal(2.5, 1.5, len(dates))
            elif indicator == "Inflation":
                data[col_name] = np.random.normal(2.0, 1.0, len(dates))
            elif indicator == "Unemployment":
                data[col_name] = np.random.normal(6.0, 2.0, len(dates))
            elif indicator == "Interest Rate":
                data[col_name] = np.random.normal(3.0, 2.0, len(dates))
            elif indicator == "Output Gap":
                data[col_name] = np.random.normal(0.0, 2.0, len(dates))
            else:
                data[col_name] = np.random.normal(0.0, 1.0, len(dates))
    
    return pd.DataFrame(data, index=dates)


def generate_mock_parameter_results() -> Dict[str, Any]:
    """Generate mock parameter estimation results."""
    regions = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    n_regions = len(regions)
    
    return {
        'regions': regions,
        'sigma': np.random.normal(1.0, 0.3, n_regions),
        'kappa': np.random.normal(0.5, 0.2, n_regions),
        'psi': np.random.normal(0.2, 0.1, n_regions),
        'phi': np.random.normal(0.3, 0.15, n_regions)
    }


def generate_mock_policy_results() -> Dict[str, Any]:
    """Generate mock policy analysis results."""
    return {
        'decomposition': {
            'total': 0.45,
            'information': 0.20,
            'weights': 0.15,
            'parameters': 0.08,
            'inflation': 0.02
        }
    }


def generate_mock_counterfactual_results() -> Dict[str, Any]:
    """Generate mock counterfactual results."""
    return {
        'welfare_outcomes': {
            'baseline': -2.45,
            'perfect_information': -1.89,
            'optimal_regional': -2.12,
            'perfect_regional': -1.56
        }
    }


# Additional visualization functions (simplified implementations)

def render_area_chart(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render area chart."""
    st.info("Area chart visualization would be implemented here")


def render_scatter_plot(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render scatter plot."""
    st.info("Scatter plot visualization would be implemented here")


def render_box_plot(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render box plot."""
    st.info("Box plot visualization would be implemented here")


def render_correlation_heatmap(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render correlation heatmap."""
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_analysis(data: Optional[pd.DataFrame]):
    """Render correlation analysis."""
    if data is not None:
        st.info("Correlation analysis results would be displayed here")
    else:
        st.warning("No data available for correlation analysis")


def render_trend_analysis(data: Optional[pd.DataFrame]):
    """Render trend analysis."""
    st.info("Trend analysis results would be displayed here")


def render_seasonality_analysis(data: Optional[pd.DataFrame]):
    """Render seasonality analysis."""
    st.info("Seasonality analysis results would be displayed here")


def render_volatility_analysis(data: Optional[pd.DataFrame]):
    """Render volatility analysis."""
    st.info("Volatility analysis results would be displayed here")


def render_regional_parameter_comparison(param_results: Dict[str, Any], parameter_focus: str):
    """Render regional parameter comparison."""
    st.info("Regional parameter comparison would be implemented here")


def render_parameter_confidence_intervals(param_results: Dict[str, Any], parameter_focus: str):
    """Render parameter confidence intervals."""
    st.info("Parameter confidence intervals would be implemented here")


def render_parameter_relationships(param_results: Dict[str, Any]):
    """Render parameter relationships."""
    st.info("Parameter relationships visualization would be implemented here")


def render_parameter_diagnostics(param_results: Dict[str, Any]):
    """Render parameter diagnostics."""
    st.info("Parameter diagnostics would be implemented here")


def render_fed_weights_chart(policy_results: Dict[str, Any]):
    """Render Fed weights chart."""
    st.info("Fed weights visualization would be implemented here")


def render_policy_mistakes_timeseries(policy_results: Dict[str, Any], time_focus: str):
    """Render policy mistakes time series."""
    st.info("Policy mistakes time series would be implemented here")


def render_regional_policy_impact(policy_results: Dict[str, Any]):
    """Render regional policy impact."""
    st.info("Regional policy impact visualization would be implemented here")


def render_welfare_loss_analysis(policy_results: Dict[str, Any]):
    """Render welfare loss analysis."""
    st.info("Welfare loss analysis would be implemented here")


def render_scenario_comparison_chart(cf_results: Dict[str, Any], scenario_focus: List[str]):
    """Render scenario comparison chart."""
    st.info("Scenario comparison chart would be implemented here")


def render_policy_rate_paths(cf_results: Dict[str, Any], scenario_focus: List[str]):
    """Render policy rate paths."""
    st.info("Policy rate paths visualization would be implemented here")


def render_regional_welfare_gains(cf_results: Dict[str, Any]):
    """Render regional welfare gains."""
    st.info("Regional welfare gains visualization would be implemented here")


def render_counterfactual_decomposition(cf_results: Dict[str, Any]):
    """Render counterfactual decomposition."""
    st.info("Counterfactual decomposition would be implemented here")


def render_session_data_visualizer():
    """Render session data visualizer."""
    st.info("Session data visualizer would be implemented here")


def render_file_upload_visualizer():
    """Render file upload visualizer."""
    st.info("File upload visualizer would be implemented here")


def render_manual_input_visualizer():
    """Render manual input visualizer."""
    st.info("Manual input visualizer would be implemented here")


def render_demo_data_visualizer():
    """Render demo data visualizer."""
    st.info("Demo data visualizer would be implemented here")