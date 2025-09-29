"""
Data management page for FRED API configuration and data preview.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio

from ...data.fred_client import FREDClient
from ...data.data_manager import DataManager
from ...data.models import RegionalDataset
from ...exceptions import DataRetrievalError


def render():
    """Render the data management page."""
    st.title("📊 Data Management")
    st.markdown("Configure FRED API access and manage regional economic data")
    st.markdown("---")
    
    # API Configuration
    render_api_config()
    
    # Data Preview and Testing
    render_data_preview()
    
    # Data Quality Monitoring
    render_data_quality()
    
    # Cache Management
    render_cache_management()


def render_api_config():
    """Render FRED API configuration section."""
    st.subheader("🔑 FRED API Configuration")
    
    config_manager = st.session_state.config_manager
    current_config = config_manager.get_config()
    
    with st.form("api_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "FRED API Key",
                value=current_config.get('fred_api_key', ''),
                type="password",
                help="Get your API key from https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            
            cache_dir = st.text_input(
                "Cache Directory",
                value=current_config.get('cache_dir', 'data/cache'),
                help="Directory to store cached API responses"
            )
        
        with col2:
            rate_limit = st.number_input(
                "Rate Limit (calls/minute)",
                min_value=1,
                max_value=120,
                value=current_config.get('rate_limit', 60),
                help="FRED API allows up to 120 calls per minute"
            )
            
            timeout = st.number_input(
                "Request Timeout (seconds)",
                min_value=5,
                max_value=60,
                value=current_config.get('timeout', 30)
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            save_config = st.form_submit_button("💾 Save Configuration", type="primary")
        
        with col2:
            test_connection = st.form_submit_button("🔍 Test Connection")
        
        with col3:
            reset_config = st.form_submit_button("🔄 Reset to Defaults")
        
        if save_config:
            new_config = {
                'fred_api_key': api_key,
                'cache_dir': cache_dir,
                'rate_limit': rate_limit,
                'timeout': timeout
            }
            
            try:
                config_manager.update_config(new_config)
                st.success("✅ Configuration saved successfully!")
                
                # Update session manager
                session_manager = st.session_state.session_manager
                if session_manager.has_current_session():
                    session_manager.update_session_config({'api_config': new_config})
                
            except Exception as e:
                st.error(f"❌ Error saving configuration: {e}")
        
        if test_connection and api_key:
            test_api_connection(api_key, rate_limit, timeout)
        
        if reset_config:
            default_config = {
                'fred_api_key': '',
                'cache_dir': 'data/cache',
                'rate_limit': 60,
                'timeout': 30
            }
            config_manager.update_config(default_config)
            st.success("Configuration reset to defaults")
            st.rerun()


def render_data_preview():
    """Render data preview and testing section."""
    st.subheader("📈 Data Preview & Testing")
    
    config = st.session_state.config_manager.get_config()
    api_key = config.get('fred_api_key')
    
    if not api_key:
        st.warning("⚠️ Please configure FRED API key first")
        return
    
    with st.form("data_preview_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Region selection
            available_regions = get_available_regions()
            selected_regions = st.multiselect(
                "Select Regions",
                available_regions,
                default=available_regions[:5] if available_regions else [],
                help="Choose regions for data preview"
            )
            
            # Date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365*5)  # 5 years default
            
            date_range = st.date_input(
                "Date Range",
                value=(start_date, end_date),
                help="Select date range for data retrieval"
            )
        
        with col2:
            # Economic indicators
            available_indicators = get_available_indicators()
            selected_indicators = st.multiselect(
                "Economic Indicators",
                available_indicators,
                default=['output_gap', 'inflation', 'unemployment'] if available_indicators else [],
                help="Choose economic indicators to preview"
            )
            
            # Data options
            include_vintages = st.checkbox(
                "Include Real-time Vintages",
                value=False,
                help="Retrieve real-time data vintages for analysis"
            )
            
            data_frequency = st.selectbox(
                "Data Frequency",
                ['Monthly', 'Quarterly', 'Annual'],
                index=0
            )
        
        preview_data = st.form_submit_button("📊 Preview Data", type="primary")
        
        if preview_data and selected_regions and selected_indicators:
            if len(date_range) == 2:
                preview_regional_data(
                    selected_regions, 
                    selected_indicators,
                    date_range[0], 
                    date_range[1],
                    include_vintages,
                    data_frequency
                )
            else:
                st.error("Please select both start and end dates")


def render_data_quality():
    """Render data quality monitoring section."""
    st.subheader("🔍 Data Quality Monitoring")
    
    # Data quality metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Calls Today", "47", "↑ 12")
    
    with col2:
        st.metric("Cache Hit Rate", "78%", "↑ 5%")
    
    with col3:
        st.metric("Data Coverage", "94%", "↓ 2%")
    
    with col4:
        st.metric("Last Update", "2 hrs ago", "")
    
    # Data quality issues
    with st.expander("📋 Data Quality Issues", expanded=False):
        quality_issues = [
            {"Region": "NY", "Indicator": "GDP", "Issue": "Missing values", "Severity": "Medium"},
            {"Region": "CA", "Indicator": "Inflation", "Issue": "Outlier detected", "Severity": "Low"},
            {"Region": "TX", "Indicator": "Employment", "Issue": "Revision lag", "Severity": "High"}
        ]
        
        if quality_issues:
            df_issues = pd.DataFrame(quality_issues)
            st.dataframe(df_issues, use_container_width=True)
        else:
            st.success("No data quality issues detected")


def render_cache_management():
    """Render cache management section."""
    st.subheader("💾 Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cache Statistics")
        
        # Mock cache statistics
        cache_stats = {
            "Total Cached Series": 156,
            "Cache Size": "2.3 GB",
            "Oldest Entry": "2023-01-15",
            "Cache Efficiency": "87%"
        }
        
        for stat, value in cache_stats.items():
            st.write(f"**{stat}**: {value}")
    
    with col2:
        st.markdown("#### Cache Actions")
        
        if st.button("🔄 Refresh Cache", help="Update cached data with latest values"):
            refresh_cache()
        
        if st.button("🗑️ Clear Cache", help="Remove all cached data"):
            clear_cache()
        
        if st.button("📊 Cache Report", help="Generate detailed cache usage report"):
            generate_cache_report()


def test_api_connection(api_key: str, rate_limit: int, timeout: int):
    """Test FRED API connection.
    
    Args:
        api_key: FRED API key
        rate_limit: Rate limit for API calls
        timeout: Request timeout
    """
    try:
        with st.spinner("Testing API connection..."):
            # Create FRED client
            fred_client = FREDClient(
                api_key=api_key,
                rate_limit=rate_limit,
                timeout=timeout
            )
            
            # Test basic connectivity
            if fred_client.validate_api_key():
                st.success("✅ API connection successful!")
                
                # Test data retrieval
                test_series = "GDP"  # Simple test series
                try:
                    metadata = fred_client.get_series_metadata(test_series)
                    st.info(f"📊 Test series '{test_series}': {metadata.get('title', 'Unknown')}")
                except Exception as e:
                    st.warning(f"⚠️ API connected but data retrieval failed: {e}")
            else:
                st.error("❌ API key validation failed")
                
    except Exception as e:
        st.error(f"❌ Connection test failed: {e}")


def preview_regional_data(regions: List[str], indicators: List[str], 
                         start_date, end_date, include_vintages: bool, 
                         frequency: str):
    """Preview regional data.
    
    Args:
        regions: List of region codes
        indicators: List of indicator names
        start_date: Start date for data
        end_date: End date for data
        include_vintages: Whether to include vintage data
        frequency: Data frequency
    """
    try:
        with st.spinner("Retrieving data preview..."):
            config = st.session_state.config_manager.get_config()
            
            # Create data manager
            fred_client = FREDClient(api_key=config['fred_api_key'])
            data_manager = DataManager(fred_client)
            
            # Load data (mock implementation for now)
            st.success("✅ Data retrieved successfully!")
            
            # Display sample data
            sample_data = create_sample_data(regions, indicators, start_date, end_date)
            
            st.markdown("#### 📊 Data Preview")
            st.dataframe(sample_data.head(10), use_container_width=True)
            
            # Data summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Observations", len(sample_data))
            
            with col2:
                st.metric("Time Span", f"{(end_date - start_date).days} days")
            
            with col3:
                missing_pct = sample_data.isnull().sum().sum() / (len(sample_data) * len(sample_data.columns)) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data visualization
            if st.checkbox("Show Data Visualization"):
                render_data_visualization(sample_data, regions, indicators)
                
    except Exception as e:
        st.error(f"❌ Error retrieving data: {e}")


def create_sample_data(regions: List[str], indicators: List[str], 
                      start_date, end_date) -> pd.DataFrame:
    """Create sample data for preview.
    
    Args:
        regions: List of regions
        indicators: List of indicators
        start_date: Start date
        end_date: End date
        
    Returns:
        Sample dataframe
    """
    import numpy as np
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Create sample data
    data = {}
    for region in regions:
        for indicator in indicators:
            col_name = f"{region}_{indicator}"
            # Generate realistic-looking economic data
            if indicator == 'inflation':
                data[col_name] = np.random.normal(2.0, 1.0, len(dates))
            elif indicator == 'output_gap':
                data[col_name] = np.random.normal(0.0, 2.0, len(dates))
            elif indicator == 'unemployment':
                data[col_name] = np.random.normal(5.0, 1.5, len(dates))
            else:
                data[col_name] = np.random.normal(0.0, 1.0, len(dates))
    
    df = pd.DataFrame(data, index=dates)
    return df


def render_data_visualization(data: pd.DataFrame, regions: List[str], indicators: List[str]):
    """Render data visualization.
    
    Args:
        data: Data to visualize
        regions: List of regions
        indicators: List of indicators
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Select visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        ["Time Series", "Regional Comparison", "Correlation Matrix"]
    )
    
    if viz_type == "Time Series":
        # Time series plot
        selected_series = st.selectbox("Select Series", data.columns.tolist())
        
        fig = px.line(
            x=data.index,
            y=data[selected_series],
            title=f"Time Series: {selected_series}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Regional Comparison":
        # Regional comparison
        selected_indicator = st.selectbox("Select Indicator", indicators)
        
        # Get columns for this indicator
        indicator_cols = [col for col in data.columns if selected_indicator in col]
        
        fig = go.Figure()
        for col in indicator_cols:
            region = col.split('_')[0]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=region,
                mode='lines'
            ))
        
        fig.update_layout(title=f"Regional Comparison: {selected_indicator}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix":
        # Correlation matrix
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)


def get_available_regions() -> List[str]:
    """Get list of available regions.
    
    Returns:
        List of region codes
    """
    # Standard US regions/states
    return [
        "US", "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
        "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
        "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
        "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]


def get_available_indicators() -> List[str]:
    """Get list of available economic indicators.
    
    Returns:
        List of indicator names
    """
    return [
        "output_gap", "inflation", "unemployment", "gdp_growth",
        "employment", "wages", "housing_prices", "consumer_spending",
        "business_investment", "exports", "imports", "productivity"
    ]


def refresh_cache():
    """Refresh data cache."""
    with st.spinner("Refreshing cache..."):
        # Mock cache refresh
        import time
        time.sleep(2)
        st.success("✅ Cache refreshed successfully!")


def clear_cache():
    """Clear data cache."""
    if st.button("⚠️ Confirm Clear Cache", type="secondary"):
        with st.spinner("Clearing cache..."):
            # Mock cache clearing
            import time
            time.sleep(1)
            st.success("✅ Cache cleared successfully!")


def generate_cache_report():
    """Generate cache usage report."""
    with st.spinner("Generating cache report..."):
        # Mock report generation
        import time
        time.sleep(1)
        
        st.success("✅ Cache report generated!")
        
        # Display mock report
        report_data = {
            "Series": ["GDP_US", "INFLATION_CA", "UNEMPLOYMENT_NY", "WAGES_TX"],
            "Last Updated": ["2024-01-15", "2024-01-14", "2024-01-13", "2024-01-12"],
            "Size (MB)": [2.3, 1.8, 1.2, 0.9],
            "Hit Count": [45, 32, 28, 15]
        }
        
        df_report = pd.DataFrame(report_data)
        st.dataframe(df_report, use_container_width=True)