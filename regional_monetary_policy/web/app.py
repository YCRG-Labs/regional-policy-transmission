"""
Main Streamlit application for Regional Monetary Policy Analysis System.

Provides interactive web interface for parameter estimation, policy analysis,
and counterfactual evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import traceback

from .session_manager import SessionManager
from .workflow_manager import WorkflowManager, WorkflowType
from .pages import (
    home_page, data_page, estimation_page, policy_page, 
    counterfactual_page, visualization_page, results_page, session_page
)
from ..config.config_manager import ConfigManager
from ..exceptions import RegionalMonetaryPolicyError
from ..logging_config import setup_logging


# Configure Streamlit page
st.set_page_config(
    page_title="Regional Monetary Policy Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logger = setup_logging()


def initialize_app():
    """Initialize application components."""
    try:
        # Initialize managers
        if 'session_manager' not in st.session_state:
            st.session_state.session_manager = SessionManager()
        
        if 'workflow_manager' not in st.session_state:
            st.session_state.workflow_manager = WorkflowManager()
        
        if 'config_manager' not in st.session_state:
            st.session_state.config_manager = ConfigManager()
        
        # Initialize progress tracking
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = {}
        
        # Initialize error state
        if 'error_state' not in st.session_state:
            st.session_state.error_state = None
            
    except Exception as e:
        st.error(f"Error initializing application: {e}")
        logger.error(f"App initialization error: {e}")
        st.stop()


def render_sidebar():
    """Render application sidebar."""
    with st.sidebar:
        st.title("🏦 Regional Monetary Policy")
        st.markdown("---")
        
        # Session management
        session_manager = st.session_state.session_manager
        current_session = session_manager.get_current_session()
        
        if current_session:
            st.success(f"📊 Session: {current_session.name}")
            st.caption(f"Status: {current_session.status}")
            
            if st.button("💾 Save Session"):
                if session_manager.save_current_session():
                    st.success("Session saved!")
                else:
                    st.error("Failed to save session")
        else:
            st.info("No active session")
        
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Home": "home",
            "📊 Data Management": "data",
            "🔢 Parameter Estimation": "estimation", 
            "📈 Policy Analysis": "policy",
            "🔄 Counterfactual Analysis": "counterfactual",
            "📋 Visualization": "visualization",
            "📄 Results & Reports": "results",
            "⚙️ Session Management": "session"
        }
        
        selected_page = st.selectbox("Navigate to:", list(pages.keys()))
        st.session_state.current_page = pages[selected_page]
        
        st.markdown("---")
        
        # Workflow status
        workflow_manager = st.session_state.workflow_manager
        current_workflow = workflow_manager.get_current_workflow()
        
        if current_workflow:
            st.subheader("🔄 Current Workflow")
            st.write(f"**{current_workflow.name}**")
            
            progress = current_workflow.get_progress()
            st.progress(progress / 100)
            st.caption(f"Progress: {progress:.1f}%")
            
            current_step = current_workflow.get_current_step()
            if current_step:
                st.write(f"Step: {current_step.title}")
            else:
                st.success("Workflow Complete!")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Session"):
                st.session_state.show_new_session = True
        
        with col2:
            if st.button("📁 Load Session"):
                st.session_state.show_load_session = True
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        # Check API connectivity
        try:
            config = st.session_state.config_manager.get_config()
            if config.get('fred_api_key'):
                st.success("✅ FRED API Configured")
            else:
                st.warning("⚠️ FRED API Key Missing")
        except Exception:
            st.error("❌ Configuration Error")


def render_main_content():
    """Render main content area based on selected page."""
    page = st.session_state.get('current_page', 'home')
    
    try:
        if page == 'home':
            home_page.render()
        elif page == 'data':
            data_page.render()
        elif page == 'estimation':
            estimation_page.render()
        elif page == 'policy':
            policy_page.render()
        elif page == 'counterfactual':
            counterfactual_page.render()
        elif page == 'visualization':
            visualization_page.render()
        elif page == 'results':
            results_page.render()
        elif page == 'session':
            session_page.render()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        logger.error(f"Page rendering error: {e}")
        
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def handle_modals():
    """Handle modal dialogs."""
    session_manager = st.session_state.session_manager
    
    # New session modal
    if st.session_state.get('show_new_session', False):
        with st.container():
            st.subheader("Create New Session")
            
            with st.form("new_session_form"):
                name = st.text_input("Session Name", placeholder="My Analysis")
                description = st.text_area("Description", placeholder="Optional description")
                
                col1, col2 = st.columns(2)
                with col1:
                    create = st.form_submit_button("Create", type="primary")
                with col2:
                    cancel = st.form_submit_button("Cancel")
                
                if create and name:
                    session = session_manager.create_session(name, description)
                    st.success(f"Created session: {session.name}")
                    st.session_state.show_new_session = False
                    st.rerun()
                
                if cancel:
                    st.session_state.show_new_session = False
                    st.rerun()
    
    # Load session modal
    if st.session_state.get('show_load_session', False):
        with st.container():
            st.subheader("Load Existing Session")
            
            sessions = session_manager.get_session_history()
            
            if not sessions:
                st.info("No saved sessions found")
                if st.button("Close"):
                    st.session_state.show_load_session = False
                    st.rerun()
            else:
                session_options = {
                    f"{s.name} ({s.last_modified.strftime('%Y-%m-%d %H:%M')})": s.session_id 
                    for s in sessions
                }
                
                selected = st.selectbox("Select Session", list(session_options.keys()))
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", type="primary"):
                        session_id = session_options[selected]
                        if session_manager.load_session(session_id):
                            st.success("Session loaded!")
                            st.session_state.show_load_session = False
                            st.rerun()
                        else:
                            st.error("Failed to load session")
                
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_load_session = False
                        st.rerun()


def render_progress_monitor():
    """Render real-time progress monitoring."""
    if 'progress_data' in st.session_state and st.session_state.progress_data:
        with st.container():
            st.subheader("🔄 Analysis Progress")
            
            for task_name, progress_info in st.session_state.progress_data.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{task_name}**")
                    progress = progress_info.get('progress', 0)
                    st.progress(progress / 100)
                
                with col2:
                    status = progress_info.get('status', 'running')
                    if status == 'completed':
                        st.success("✅")
                    elif status == 'error':
                        st.error("❌")
                    else:
                        st.info("⏳")
                
                if progress_info.get('message'):
                    st.caption(progress_info['message'])


def main():
    """Main application entry point."""
    # Initialize application
    initialize_app()
    
    # Handle any startup errors
    if st.session_state.get('error_state'):
        st.error(f"Application Error: {st.session_state.error_state}")
        if st.button("Clear Error"):
            st.session_state.error_state = None
            st.rerun()
        return
    
    # Render UI components
    render_sidebar()
    
    # Handle modals first
    handle_modals()
    
    # Render main content if no modals are active
    if not (st.session_state.get('show_new_session') or 
            st.session_state.get('show_load_session')):
        render_main_content()
    
    # Render progress monitor
    render_progress_monitor()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Regional Monetary Policy Analysis System | "
        "Built with Streamlit | "
        f"Session: {st.session_state.session_manager.get_current_session().name if st.session_state.session_manager.has_current_session() else 'None'}"
    )


if __name__ == "__main__":
    main()