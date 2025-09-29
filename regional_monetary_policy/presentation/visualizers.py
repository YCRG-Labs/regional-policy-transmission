"""
Core visualization components for regional monetary policy analysis.

This module provides interactive visualizations for regional economic data,
parameter estimates, policy analysis, and counterfactual scenarios using
Plotly for interactive charts and maps.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any
import warnings

from ..data.models import RegionalDataset, ValidationReport
from ..econometric.models import RegionalParameters, EstimationResults
from ..policy.models import (
    PolicyScenario, PolicyMistakeComponents, 
    ComparisonResults, WelfareDecomposition
)


class RegionalMapVisualizer:
    """
    Creates interactive regional maps for economic indicators.
    
    This class handles the creation of choropleth maps showing regional
    variations in economic indicators like output gaps, inflation rates,
    and parameter estimates.
    """
    
    def __init__(self, region_codes: List[str], region_names: Optional[List[str]] = None):
        """
        Initialize the map visualizer.
        
        Args:
            region_codes: List of region identifiers (e.g., state codes)
            region_names: Optional list of human-readable region names
        """
        self.region_codes = region_codes
        self.region_names = region_names or [f"Region {i+1}" for i in range(len(region_codes))]
        
        if len(self.region_codes) != len(self.region_names):
            raise ValueError("Region codes and names must have same length")
    
    def create_indicator_map(
        self, 
        data: pd.Series, 
        title: str,
        colorscale: str = 'RdYlBu_r',
        show_colorbar: bool = True
    ) -> go.Figure:
        """
        Create an interactive map showing regional variation in an economic indicator.
        
        Args:
            data: Series with region codes as index and indicator values
            title: Title for the map
            colorscale: Plotly colorscale name
            show_colorbar: Whether to show the color scale bar
            
        Returns:
            Plotly figure with interactive map
        """
        # Prepare data for mapping
        map_data = pd.DataFrame({
            'region_code': self.region_codes,
            'region_name': self.region_names,
            'value': [data.get(code, np.nan) for code in self.region_codes]
        })
        
        # Create choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=map_data['region_code'],
            z=map_data['value'],
            locationmode='USA-states',  # Assuming US states for now
            colorscale=colorscale,
            text=map_data['region_name'],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{title}: %{{z:.3f}}<br>' +
                         '<extra></extra>',
            showscale=show_colorbar,
            colorbar=dict(title=title) if show_colorbar else None
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            ),
            width=800,
            height=500
        )
        
        return fig
    
    def create_multi_indicator_map(
        self, 
        data_dict: Dict[str, pd.Series],
        title: str = "Regional Economic Indicators"
    ) -> go.Figure:
        """
        Create a map with multiple indicators using dropdown selection.
        
        Args:
            data_dict: Dictionary mapping indicator names to data series
            title: Overall title for the visualization
            
        Returns:
            Plotly figure with dropdown selector for different indicators
        """
        # Create base figure with first indicator
        first_indicator = list(data_dict.keys())[0]
        fig = self.create_indicator_map(
            data_dict[first_indicator], 
            first_indicator
        )
        
        # Create dropdown buttons for other indicators
        buttons = []
        for indicator_name, indicator_data in data_dict.items():
            map_data = pd.DataFrame({
                'region_code': self.region_codes,
                'value': [indicator_data.get(code, np.nan) for code in self.region_codes]
            })
            
            button = dict(
                label=indicator_name,
                method='restyle',
                args=[{
                    'z': [map_data['value']],
                    'hovertemplate': f'<b>%{{text}}</b><br>{indicator_name}: %{{z:.3f}}<br><extra></extra>',
                    'colorbar.title': indicator_name
                }]
            )
            buttons.append(button)
        
        # Add dropdown menu
        fig.update_layout(
            title=title,
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                )
            ]
        )
        
        return fig


class TimeSeriesVisualizer:
    """
    Creates time series visualizations for policy transmission and spillover effects.
    """
    
    def __init__(self):
        """Initialize the time series visualizer."""
        pass
    
    def create_policy_transmission_plot(
        self,
        policy_rates: pd.Series,
        regional_outcomes: pd.DataFrame,
        title: str = "Monetary Policy Transmission"
    ) -> go.Figure:
        """
        Create a plot showing how policy rates affect regional outcomes.
        
        Args:
            policy_rates: Time series of policy interest rates
            regional_outcomes: DataFrame with regional output gaps and inflation
            title: Plot title
            
        Returns:
            Plotly figure with policy transmission visualization
        """
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Policy Rate and Regional Output Gaps', 'Regional Inflation Rates'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Add policy rate to top subplot
        fig.add_trace(
            go.Scatter(
                x=policy_rates.index,
                y=policy_rates.values,
                name='Policy Rate',
                line=dict(color='black', width=3),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add regional output gaps
        n_regions = len(regional_outcomes.index) // 2
        colors = px.colors.qualitative.Set1[:n_regions]
        
        for i in range(n_regions):
            region_name = f"Region {i+1}"
            output_gap = regional_outcomes.iloc[i]
            
            fig.add_trace(
                go.Scatter(
                    x=output_gap.index,
                    y=output_gap.values,
                    name=f'{region_name} Output Gap',
                    line=dict(color=colors[i]),
                    opacity=0.7
                ),
                row=1, col=1, secondary_y=False
            )
        
        # Add regional inflation rates
        for i in range(n_regions):
            region_name = f"Region {i+1}"
            inflation = regional_outcomes.iloc[i + n_regions]
            
            fig.add_trace(
                go.Scatter(
                    x=inflation.index,
                    y=inflation.values,
                    name=f'{region_name} Inflation',
                    line=dict(color=colors[i], dash='dash'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Output Gap (%)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Policy Rate (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Inflation Rate (%)", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def create_spillover_effects_plot(
        self,
        regional_data: RegionalDataset,
        spatial_weights: np.ndarray,
        title: str = "Regional Spillover Effects"
    ) -> go.Figure:
        """
        Visualize spatial spillover effects between regions.
        
        Args:
            regional_data: Regional economic dataset
            spatial_weights: Spatial weight matrix
            title: Plot title
            
        Returns:
            Plotly figure showing spillover patterns
        """
        # Calculate spatial lags for output gaps
        output_gaps = regional_data.output_gaps.values
        spatial_lags = spatial_weights @ output_gaps
        
        # Create correlation matrix between regions and their spatial lags
        correlations = np.corrcoef(output_gaps, spatial_lags)[:len(output_gaps), len(output_gaps):]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            x=[f"Spatial Lag {i+1}" for i in range(len(correlations))],
            y=[f"Region {i+1}" for i in range(len(correlations))],
            colorscale='RdBu',
            zmid=0,
            text=correlations,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hovertemplate='Region %{y}<br>Spatial Lag %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Spatial Lag Components",
            yaxis_title="Regions",
            width=600,
            height=500
        )
        
        return fig


class ParameterVisualizer:
    """
    Creates visualizations for parameter estimation results.
    """
    
    def __init__(self):
        """Initialize the parameter visualizer."""
        pass
    
    def create_parameter_estimates_plot(
        self,
        regional_params: RegionalParameters,
        title: str = "Regional Parameter Estimates"
    ) -> go.Figure:
        """
        Create a comprehensive plot of parameter estimates with confidence intervals.
        
        Args:
            regional_params: Estimated regional parameters
            title: Plot title
            
        Returns:
            Plotly figure with parameter estimates and confidence intervals
        """
        n_regions = regional_params.n_regions
        regions = [f"Region {i+1}" for i in range(n_regions)]
        
        # Create subplots for each parameter
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Interest Rate Sensitivity (σ)', 'Phillips Curve Slope (κ)', 
                          'Demand Spillover (ψ)', 'Price Spillover (φ)', 'Discount Factor (β)', ''),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        parameters = {
            'sigma': (regional_params.sigma, 1, 1),
            'kappa': (regional_params.kappa, 1, 2), 
            'psi': (regional_params.psi, 2, 1),
            'phi': (regional_params.phi, 2, 2),
            'beta': (regional_params.beta, 2, 3)
        }
        
        colors = px.colors.qualitative.Set1[:n_regions]
        
        for param_name, (values, row, col) in parameters.items():
            # Add point estimates
            fig.add_trace(
                go.Scatter(
                    x=regions,
                    y=values,
                    mode='markers+lines',
                    name=param_name,
                    marker=dict(size=8, color=colors),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add confidence intervals if available
            if param_name in regional_params.confidence_intervals:
                lower, upper = regional_params.confidence_intervals[param_name]
                
                fig.add_trace(
                    go.Scatter(
                        x=regions + regions[::-1],
                        y=np.concatenate([upper, lower[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% CI',
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        # Update y-axis labels
        for i, (param_name, (_, row, col)) in enumerate(parameters.items()):
            fig.update_yaxes(title_text=f"{param_name}", row=row, col=col)
        
        return fig
    
    def create_parameter_comparison_table(
        self,
        regional_params: RegionalParameters,
        title: str = "Parameter Estimates Summary"
    ) -> go.Figure:
        """
        Create a table summarizing parameter estimates across regions.
        
        Args:
            regional_params: Estimated regional parameters
            title: Table title
            
        Returns:
            Plotly figure with parameter summary table
        """
        # Get parameter summary DataFrame
        summary_df = regional_params.get_parameter_summary()
        
        # Prepare table data
        header_values = ['Region'] + list(summary_df.columns)
        cell_values = [summary_df.index.tolist()]
        
        for col in summary_df.columns:
            if summary_df[col].dtype in [np.float64, np.float32]:
                cell_values.append([f"{val:.4f}" for val in summary_df[col]])
            else:
                cell_values.append(summary_df[col].tolist())
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=cell_values,
                fill_color='lavender',
                align='center',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=title,
            width=1000,
            height=400
        )
        
        return fig


class PolicyAnalysisVisualizer:
    """
    Creates visualizations for policy mistake decomposition and analysis.
    """
    
    def __init__(self):
        """Initialize the policy analysis visualizer."""
        pass
    
    def create_mistake_decomposition_plot(
        self,
        mistake_components: PolicyMistakeComponents,
        title: str = "Policy Mistake Decomposition"
    ) -> go.Figure:
        """
        Create a visualization of policy mistake decomposition.
        
        Args:
            mistake_components: Policy mistake decomposition results
            title: Plot title
            
        Returns:
            Plotly figure with mistake decomposition
        """
        components = [
            'Information<br>Effect',
            'Weight<br>Misallocation', 
            'Parameter<br>Misspecification',
            'Inflation<br>Response'
        ]
        
        values = [
            mistake_components.information_effect,
            mistake_components.weight_misallocation_effect,
            mistake_components.parameter_misspecification_effect,
            mistake_components.inflation_response_effect
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        # Create subplot with bar chart and pie chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Absolute Contributions', 'Relative Contributions'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart of absolute contributions
        fig.add_trace(
            go.Bar(
                x=components,
                y=values,
                marker_color=colors,
                text=[f"{v:.4f}" for v in values],
                textposition='outside',
                name='Absolute'
            ),
            row=1, col=1
        )
        
        # Pie chart of relative contributions (absolute values)
        abs_values = [abs(v) for v in values]
        if sum(abs_values) > 0:
            fig.add_trace(
                go.Pie(
                    labels=components,
                    values=abs_values,
                    marker_colors=colors,
                    name='Relative'
                ),
                row=1, col=2
            )
        
        # Add horizontal line at zero for bar chart only
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(components)-0.5,
            y0=0, y1=0,
            line=dict(dash="dash", color="black", width=1),
            opacity=0.5,
            row=1, col=1
        )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Contribution (pp)", row=1, col=1)
        
        return fig
    
    def create_mistake_time_series(
        self,
        mistake_history: pd.DataFrame,
        title: str = "Policy Mistakes Over Time"
    ) -> go.Figure:
        """
        Create time series plot of policy mistakes and their components.
        
        Args:
            mistake_history: DataFrame with mistake components over time
            title: Plot title
            
        Returns:
            Plotly figure with time series of mistakes
        """
        fig = go.Figure()
        
        # Add total mistake
        fig.add_trace(
            go.Scatter(
                x=mistake_history.index,
                y=mistake_history['total_mistake'],
                name='Total Mistake',
                line=dict(color='black', width=3)
            )
        )
        
        # Add components
        components = {
            'information_effect': ('Information Effect', 'blue'),
            'weight_misallocation_effect': ('Weight Misallocation', 'red'),
            'parameter_misspecification_effect': ('Parameter Misspecification', 'green'),
            'inflation_response_effect': ('Inflation Response', 'orange')
        }
        
        for col, (name, color) in components.items():
            if col in mistake_history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=mistake_history.index,
                        y=mistake_history[col],
                        name=name,
                        line=dict(color=color, dash='dash'),
                        opacity=0.7
                    )
                )
        
        # Add horizontal line at zero
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Policy Mistake (percentage points)",
            hovermode='x unified',
            height=500
        )
        
        return fig


class CounterfactualVisualizer:
    """
    Creates visualizations for counterfactual scenario comparisons.
    """
    
    def __init__(self):
        """Initialize the counterfactual visualizer."""
        pass
    
    def create_scenario_comparison_plot(
        self,
        scenarios: List[PolicyScenario],
        title: str = "Counterfactual Policy Scenarios"
    ) -> go.Figure:
        """
        Create a comparison plot of different policy scenarios.
        
        Args:
            scenarios: List of policy scenarios to compare
            title: Plot title
            
        Returns:
            Plotly figure comparing scenarios
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Policy Rates', 'Welfare Outcomes'),
            vertical_spacing=0.15
        )
        
        colors = px.colors.qualitative.Set1[:len(scenarios)]
        
        # Plot policy rates
        for i, scenario in enumerate(scenarios):
            fig.add_trace(
                go.Scatter(
                    x=scenario.policy_rates.index,
                    y=scenario.policy_rates.values,
                    name=scenario.name,
                    line=dict(color=colors[i], width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot welfare outcomes as bar chart
        welfare_values = [scenario.welfare_outcome for scenario in scenarios]
        scenario_names = [scenario.name for scenario in scenarios]
        
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=welfare_values,
                marker_color=colors,
                text=[f"{w:.6f}" for w in welfare_values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Policy Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Welfare", row=2, col=1)
        
        return fig
    
    def create_welfare_comparison_table(
        self,
        comparison_results: ComparisonResults,
        title: str = "Welfare Comparison Results"
    ) -> go.Figure:
        """
        Create a table comparing welfare outcomes across scenarios.
        
        Args:
            comparison_results: Results from scenario comparison
            title: Table title
            
        Returns:
            Plotly figure with welfare comparison table
        """
        # Prepare table data
        summary_df = comparison_results.summary_table()
        
        header_values = ['Scenario', 'Welfare', 'Ranking']
        cell_values = [
            summary_df['Scenario'].tolist(),
            [f"{w:.6f}" for w in summary_df['Welfare']],
            summary_df['Ranking'].tolist()
        ]
        
        # Color code by ranking
        cell_colors = []
        for ranking in summary_df['Ranking']:
            if ranking == 1:
                cell_colors.append('lightgreen')
            elif ranking == 2:
                cell_colors.append('lightblue')
            elif ranking == 3:
                cell_colors.append('lightyellow')
            else:
                cell_colors.append('lightcoral')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=cell_values,
                fill_color=[['white']*len(summary_df), ['white']*len(summary_df), cell_colors],
                align='center',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title=title,
            width=600,
            height=300
        )
        
        return fig
    
    def create_regional_impact_heatmap(
        self,
        scenarios: List[PolicyScenario],
        title: str = "Regional Impact Comparison"
    ) -> go.Figure:
        """
        Create a heatmap showing regional impacts across scenarios.
        
        Args:
            scenarios: List of policy scenarios
            title: Plot title
            
        Returns:
            Plotly figure with regional impact heatmap
        """
        # Collect regional impacts for all scenarios
        impact_data = []
        scenario_names = []
        
        for scenario in scenarios:
            regional_impacts = scenario.get_regional_impacts()
            impact_data.append(regional_impacts['welfare_loss'].values)
            scenario_names.append(scenario.name)
        
        if not impact_data:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No regional impact data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create heatmap
        impact_matrix = np.array(impact_data)
        region_names = [f"Region {i+1}" for i in range(impact_matrix.shape[1])]
        
        fig = go.Figure(data=go.Heatmap(
            z=impact_matrix,
            x=region_names,
            y=scenario_names,
            colorscale='Reds',
            text=impact_matrix,
            texttemplate="%{text:.4f}",
            textfont={"size": 10},
            hovertemplate='Scenario: %{y}<br>Region: %{x}<br>Welfare Loss: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Regions",
            yaxis_title="Policy Scenarios",
            width=600,
            height=400
        )
        
        return fig
    
    def create_economic_indicators_dashboard(
        self,
        regional_data: RegionalDataset,
        title: str = "Regional Economic Indicators Dashboard"
    ) -> go.Figure:
        """
        Create a comprehensive dashboard showing multiple economic indicators.
        
        Args:
            regional_data: Regional economic dataset
            title: Dashboard title
            
        Returns:
            Plotly figure with multi-indicator dashboard
        """
        # Prepare data for multiple indicators
        indicators = {}
        
        # Average output gaps by region
        indicators['Output Gap'] = regional_data.output_gaps.mean(axis=1)
        
        # Average inflation by region
        indicators['Inflation Rate'] = regional_data.inflation_rates.mean(axis=1)
        
        # Output gap volatility by region
        indicators['Output Volatility'] = regional_data.output_gaps.std(axis=1)
        
        # Inflation volatility by region
        indicators['Inflation Volatility'] = regional_data.inflation_rates.std(axis=1)
        
        return self.create_multi_indicator_map(indicators, title)


# Add this method to RegionalMapVisualizer class
def create_economic_indicators_dashboard(
    self,
    regional_data: RegionalDataset,
    title: str = "Regional Economic Indicators Dashboard"
) -> go.Figure:
    """
    Create a comprehensive dashboard showing multiple economic indicators.
    
    Args:
        regional_data: Regional economic dataset
        title: Dashboard title
        
    Returns:
        Plotly figure with multi-indicator dashboard
    """
    # Prepare data for multiple indicators
    indicators = {}
    
    # Average output gaps by region
    indicators['Output Gap'] = regional_data.output_gaps.mean(axis=1)
    
    # Average inflation by region
    indicators['Inflation Rate'] = regional_data.inflation_rates.mean(axis=1)
    
    # Output gap volatility by region
    indicators['Output Volatility'] = regional_data.output_gaps.std(axis=1)
    
    # Inflation volatility by region
    indicators['Inflation Volatility'] = regional_data.inflation_rates.std(axis=1)
    
    return self.create_multi_indicator_map(indicators, title)

# Patch the RegionalMapVisualizer class to include this method
RegionalMapVisualizer.create_economic_indicators_dashboard = create_economic_indicators_dashboard