"""
Home page for the Regional Monetary Policy Analysis System.

Provides overview, quick start options, and system status.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from ...config.config_manager import ConfigManager
from ..workflow_manager import WorkflowType


def render():
    """Render the home page."""
    st.title("🏦 Regional Monetary Policy Analysis System")
    st.markdown("---")
    
    # Welcome message and overview
    st.markdown("""
    Welcome to the Regional Monetary Policy Analysis System! This platform provides 
    comprehensive tools for analyzing regional heterogeneity in monetary policy 
    transmission, estimating structural parameters, and evaluating policy effectiveness.
    """)
    
    # Quick start section
    render_quick_start()
    
    # System overview
    render_system_overview()
    
    # Recent activity
    render_recent_activity()
    
    # Getting started guide
    render_getting_started()


def render_quick_start():
    """Render quick start section."""
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 New Analysis")
        st.markdown("Start a new regional monetary policy analysis")
        
        workflow_manager = st.session_state.workflow_manager
        workflows = workflow_manager.get_available_workflows()
        
        workflow_names = [w.name for w in workflows]
        selected_workflow = st.selectbox("Choose Analysis Type:", workflow_names, key="quick_start_workflow")
        
        if st.button("Start Analysis", type="primary", key="start_analysis"):
            # Find selected workflow
            selected = next(w for w in workflows if w.name == selected_workflow)
            if workflow_manager.start_workflow(selected.workflow_id):
                st.success(f"Started {selected.name} workflow!")
                st.session_state.current_page = "estimation"  # Navigate to appropriate page
                st.rerun()
    
    with col2:
        st.markdown("### 📁 Load Session")
        st.markdown("Continue working on a previous analysis")
        
        session_manager = st.session_state.session_manager
        recent_sessions = session_manager.get_session_history()[:5]  # Last 5 sessions
        
        if recent_sessions:
            session_options = {
                f"{s.name} ({s.last_modified.strftime('%m/%d')})": s.session_id 
                for s in recent_sessions
            }
            
            selected_session = st.selectbox("Recent Sessions:", list(session_options.keys()), key="quick_load")
            
            if st.button("Load Session", key="load_recent"):
                session_id = session_options[selected_session]
                if session_manager.load_session(session_id):
                    st.success("Session loaded!")
                    st.rerun()
        else:
            st.info("No recent sessions found")
            if st.button("Browse All Sessions", key="browse_sessions"):
                st.session_state.current_page = "session"
                st.rerun()
    
    with col3:
        st.markdown("### 📖 Documentation")
        st.markdown("Learn about the system and methodology")
        
        doc_options = [
            "System Overview",
            "Theoretical Framework", 
            "Data Requirements",
            "Estimation Procedures",
            "Policy Analysis Methods",
            "API Documentation"
        ]
        
        selected_doc = st.selectbox("Documentation:", doc_options, key="quick_docs")
        
        if st.button("View Documentation", key="view_docs"):
            render_documentation_modal(selected_doc)


def render_system_overview():
    """Render system overview section."""
    st.subheader("📋 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Features")
        features = [
            "**Regional Parameter Estimation**: Three-stage GMM estimation of structural parameters",
            "**Spatial Modeling**: Construct and validate spatial weight matrices",
            "**Policy Analysis**: Decompose monetary policy mistakes into components",
            "**Counterfactual Analysis**: Evaluate alternative policy scenarios",
            "**Interactive Visualization**: Regional maps and time series plots",
            "**Export & Reporting**: Generate publication-ready results"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
    
    with col2:
        st.markdown("#### 🔧 System Status")
        
        # Check system components
        status_items = check_system_status()
        
        for item, status in status_items.items():
            if status['ok']:
                st.success(f"✅ {item}: {status['message']}")
            else:
                st.error(f"❌ {item}: {status['message']}")


def render_recent_activity():
    """Render recent activity section."""
    st.subheader("📈 Recent Activity")
    
    session_manager = st.session_state.session_manager
    recent_sessions = session_manager.get_session_history()[:10]
    
    if recent_sessions:
        # Create activity dataframe
        activity_data = []
        for session in recent_sessions:
            activity_data.append({
                'Session': session.name,
                'Type': session.config.get('analysis_type', 'Unknown'),
                'Status': session.status,
                'Last Modified': session.last_modified.strftime('%Y-%m-%d %H:%M'),
                'Progress': f"{len([r for r in session.results.keys()])} results"
            })
        
        df = pd.DataFrame(activity_data)
        st.dataframe(df, use_container_width=True)
        
        # Activity summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sessions = len(recent_sessions)
            st.metric("Total Sessions", total_sessions)
        
        with col2:
            completed_sessions = len([s for s in recent_sessions if s.status == 'completed'])
            st.metric("Completed", completed_sessions)
        
        with col3:
            active_sessions = len([s for s in recent_sessions if s.status == 'running'])
            st.metric("Active", active_sessions)
        
        with col4:
            recent_activity = len([s for s in recent_sessions 
                                 if s.last_modified > datetime.now() - timedelta(days=7)])
            st.metric("This Week", recent_activity)
    
    else:
        st.info("No recent activity. Start your first analysis above!")


def render_getting_started():
    """Render getting started guide."""
    with st.expander("📚 Getting Started Guide", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide
        
        #### 1. 🔑 Setup API Access
        - Obtain a FRED API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
        - Configure the API key in the Data Management page
        
        #### 2. 📊 Choose Analysis Type
        - **Parameter Estimation**: Estimate regional structural parameters
        - **Policy Analysis**: Analyze Fed policy effectiveness and mistakes
        - **Counterfactual Analysis**: Evaluate alternative policy scenarios
        
        #### 3. 🔄 Follow Guided Workflow
        - Each analysis type has a structured workflow
        - Complete each step before proceeding to the next
        - Save your progress regularly
        
        #### 4. 📈 Review Results
        - Visualize results with interactive charts and maps
        - Export data and generate reports
        - Compare different scenarios and specifications
        
        ### 💡 Tips for Success
        - Start with a small region set for initial testing
        - Use recent data periods for better API performance
        - Save intermediate results to avoid re-computation
        - Check diagnostics and robustness of estimates
        """)


def render_documentation_modal(doc_type: str):
    """Render documentation modal.
    
    Args:
        doc_type: Type of documentation to display
    """
    st.session_state.show_documentation = doc_type
    
    # This would be handled in the main app with a modal
    st.info(f"Documentation for {doc_type} would be displayed here")


def check_system_status() -> Dict[str, Dict[str, Any]]:
    """Check system component status.
    
    Returns:
        Dictionary of component statuses
    """
    status = {}
    
    try:
        # Check configuration
        config_manager = st.session_state.config_manager
        config = config_manager.get_config()
        
        if config.get('fred_api_key'):
            status['FRED API'] = {'ok': True, 'message': 'Configured'}
        else:
            status['FRED API'] = {'ok': False, 'message': 'API key missing'}
        
        # Check data directory
        import os
        if os.path.exists('data'):
            status['Data Directory'] = {'ok': True, 'message': 'Available'}
        else:
            status['Data Directory'] = {'ok': False, 'message': 'Not found'}
        
        # Check session storage
        session_manager = st.session_state.session_manager
        if os.path.exists(session_manager.sessions_dir):
            status['Session Storage'] = {'ok': True, 'message': 'Available'}
        else:
            status['Session Storage'] = {'ok': False, 'message': 'Not initialized'}
        
        # Check workflow manager
        workflow_manager = st.session_state.workflow_manager
        workflows = workflow_manager.get_available_workflows()
        status['Workflows'] = {'ok': True, 'message': f'{len(workflows)} available'}
        
    except Exception as e:
        status['System'] = {'ok': False, 'message': f'Error: {str(e)}'}
    
    return status