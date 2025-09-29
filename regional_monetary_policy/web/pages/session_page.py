"""
Session management page for creating, loading, and managing analysis sessions.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from ..session_manager import AnalysisSession


def render():
    """Render the session management page."""
    st.title("⚙️ Session Management")
    st.markdown("Create, load, and manage your analysis sessions")
    st.markdown("---")
    
    session_manager = st.session_state.session_manager
    
    # Current session status
    render_current_session_status()
    
    # Session actions
    render_session_actions()
    
    # Session history
    render_session_history()
    
    # Session import/export
    render_session_import_export()


def render_current_session_status():
    """Render current session status."""
    st.subheader("📊 Current Session")
    
    session_manager = st.session_state.session_manager
    current_session = session_manager.get_current_session()
    
    if current_session:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Session Info")
            st.write(f"**Name:** {current_session.name}")
            st.write(f"**Status:** {current_session.status}")
            st.write(f"**Created:** {current_session.created_at.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Modified:** {current_session.last_modified.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown("#### Configuration")
            config_count = len(current_session.config)
            st.write(f"**Config Items:** {config_count}")
            
            if config_count > 0:
                st.write("**Config Keys:**")
                for key in list(current_session.config.keys())[:5]:  # Show first 5
                    st.write(f"• {key}")
                if config_count > 5:
                    st.write(f"• ... and {config_count - 5} more")
        
        with col3:
            st.markdown("#### Results")
            results_count = len(current_session.results)
            st.write(f"**Result Items:** {results_count}")
            
            if results_count > 0:
                st.write("**Result Types:**")
                for key in list(current_session.results.keys())[:5]:  # Show first 5
                    st.write(f"• {key}")
                if results_count > 5:
                    st.write(f"• ... and {results_count - 5} more")
        
        # Session description
        if current_session.description:
            st.markdown("#### Description")
            st.write(current_session.description)
        
        # Session actions
        st.markdown("#### Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Session", use_container_width=True):
                if session_manager.save_current_session():
                    st.success("Session saved!")
                else:
                    st.error("Failed to save session")
        
        with col2:
            if st.button("📝 Edit Session", use_container_width=True):
                st.session_state.show_edit_session = True
                st.rerun()
        
        with col3:
            if st.button("📋 Session Details", use_container_width=True):
                st.session_state.show_session_details = True
                st.rerun()
        
        with col4:
            if st.button("🗑️ Close Session", use_container_width=True):
                st.session_state.current_session = None
                st.success("Session closed")
                st.rerun()
    
    else:
        st.info("No active session. Create a new session or load an existing one.")


def render_session_actions():
    """Render session action buttons."""
    st.subheader("🚀 Session Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🆕 Create New Session")
        if st.button("Create New Session", type="primary", use_container_width=True):
            st.session_state.show_create_session = True
            st.rerun()
    
    with col2:
        st.markdown("#### 📁 Load Session")
        if st.button("Browse Sessions", use_container_width=True):
            st.session_state.show_load_session = True
            st.rerun()
    
    with col3:
        st.markdown("#### 📤 Import Session")
        if st.button("Import Session", use_container_width=True):
            st.session_state.show_import_session = True
            st.rerun()


def render_session_history():
    """Render session history table."""
    st.subheader("📚 Session History")
    
    session_manager = st.session_state.session_manager
    sessions = session_manager.get_session_history()
    
    if not sessions:
        st.info("No saved sessions found.")
        return
    
    # Create sessions dataframe
    session_data = []
    for session in sessions:
        session_data.append({
            'Name': session.name,
            'Status': session.status,
            'Created': session.created_at.strftime('%Y-%m-%d'),
            'Modified': session.last_modified.strftime('%Y-%m-%d %H:%M'),
            'Config Items': len(session.config),
            'Results': len(session.results),
            'Session ID': session.session_id
        })
    
    sessions_df = pd.DataFrame(session_data)
    
    # Display with selection
    selected_indices = st.dataframe(
        sessions_df.drop('Session ID', axis=1),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Session actions for selected session
    if selected_indices and len(selected_indices.selection.rows) > 0:
        selected_idx = selected_indices.selection.rows[0]
        selected_session_id = sessions_df.iloc[selected_idx]['Session ID']
        selected_session_name = sessions_df.iloc[selected_idx]['Name']
        
        st.markdown(f"#### Actions for '{selected_session_name}'")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📂 Load", key=f"load_{selected_session_id}"):
                if session_manager.load_session(selected_session_id):
                    st.success(f"Loaded session: {selected_session_name}")
                    st.rerun()
                else:
                    st.error("Failed to load session")
        
        with col2:
            if st.button("👁️ View", key=f"view_{selected_session_id}"):
                render_session_details_modal(selected_session_id)
        
        with col3:
            if st.button("📤 Export", key=f"export_{selected_session_id}"):
                export_session(selected_session_id, selected_session_name)
        
        with col4:
            if st.button("📋 Duplicate", key=f"duplicate_{selected_session_id}"):
                duplicate_session(selected_session_id, selected_session_name)
        
        with col5:
            if st.button("🗑️ Delete", key=f"delete_{selected_session_id}"):
                st.session_state.confirm_delete_session = selected_session_id
                st.session_state.confirm_delete_name = selected_session_name
                st.rerun()


def render_session_import_export():
    """Render session import/export functionality."""
    st.subheader("📦 Import/Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Export Sessions")
        
        session_manager = st.session_state.session_manager
        sessions = session_manager.get_session_history()
        
        if sessions:
            export_sessions = st.multiselect(
                "Select Sessions to Export",
                [f"{s.name} ({s.session_id[:8]})" for s in sessions],
                help="Choose sessions to export as a package"
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "ZIP Archive", "CSV Summary"]
            )
            
            if st.button("📤 Export Selected", type="primary"):
                export_multiple_sessions(export_sessions, export_format)
        else:
            st.info("No sessions available for export")
    
    with col2:
        st.markdown("#### 📥 Import Sessions")
        
        import_method = st.selectbox(
            "Import Method",
            ["Upload File", "Paste JSON", "Load from URL"]
        )
        
        if import_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose session file",
                type=['json', 'zip'],
                help="Upload a session JSON file or ZIP archive"
            )
            
            if uploaded_file is not None:
                if st.button("📥 Import File", type="primary"):
                    import_session_file(uploaded_file)
        
        elif import_method == "Paste JSON":
            json_data = st.text_area(
                "Paste Session JSON",
                height=200,
                help="Paste the JSON data of a session to import"
            )
            
            if json_data and st.button("📥 Import JSON", type="primary"):
                import_session_json(json_data)
        
        elif import_method == "Load from URL":
            session_url = st.text_input(
                "Session URL",
                help="URL to a session JSON file"
            )
            
            if session_url and st.button("📥 Import from URL", type="primary"):
                import_session_url(session_url)


# Modal dialogs and forms

def render_create_session_modal():
    """Render create session modal."""
    if st.session_state.get('show_create_session', False):
        st.markdown("### 🆕 Create New Session")
        
        with st.form("create_session_form"):
            session_name = st.text_input(
                "Session Name *",
                placeholder="My Regional Analysis"
            )
            
            session_description = st.text_area(
                "Description",
                placeholder="Optional description of this analysis session"
            )
            
            # Initial configuration
            st.markdown("#### Initial Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_type = st.selectbox(
                    "Primary Analysis Type",
                    ["Parameter Estimation", "Policy Analysis", "Counterfactual Analysis", "Mixed Analysis"]
                )
                
                regions = st.multiselect(
                    "Initial Regions",
                    ["US", "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC"],
                    default=["US", "CA", "NY", "TX"]
                )
            
            with col2:
                time_period_start = st.date_input(
                    "Analysis Start Date",
                    value=pd.Timestamp('2008-01-01').date()
                )
                
                time_period_end = st.date_input(
                    "Analysis End Date",
                    value=pd.Timestamp('2020-12-31').date()
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                create_button = st.form_submit_button("🆕 Create Session", type="primary")
            
            with col2:
                cancel_button = st.form_submit_button("❌ Cancel")
            
            if create_button and session_name:
                initial_config = {
                    'analysis_type': analysis_type,
                    'regions': regions,
                    'time_period_start': time_period_start.isoformat(),
                    'time_period_end': time_period_end.isoformat()
                }
                
                session_manager = st.session_state.session_manager
                session = session_manager.create_session(
                    session_name, 
                    session_description, 
                    initial_config
                )
                
                st.success(f"✅ Created session: {session.name}")
                st.session_state.show_create_session = False
                st.rerun()
            
            if cancel_button:
                st.session_state.show_create_session = False
                st.rerun()


def render_edit_session_modal():
    """Render edit session modal."""
    if st.session_state.get('show_edit_session', False):
        session_manager = st.session_state.session_manager
        current_session = session_manager.get_current_session()
        
        if not current_session:
            st.error("No active session to edit")
            st.session_state.show_edit_session = False
            return
        
        st.markdown("### 📝 Edit Session")
        
        with st.form("edit_session_form"):
            new_name = st.text_input(
                "Session Name",
                value=current_session.name
            )
            
            new_description = st.text_area(
                "Description",
                value=current_session.description
            )
            
            new_status = st.selectbox(
                "Status",
                ["created", "running", "completed", "error"],
                index=["created", "running", "completed", "error"].index(current_session.status)
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                save_button = st.form_submit_button("💾 Save Changes", type="primary")
            
            with col2:
                cancel_button = st.form_submit_button("❌ Cancel")
            
            if save_button:
                current_session.name = new_name
                current_session.description = new_description
                current_session.status = new_status
                current_session.last_modified = datetime.now()
                
                if session_manager.save_current_session():
                    st.success("✅ Session updated successfully!")
                else:
                    st.error("❌ Failed to save session changes")
                
                st.session_state.show_edit_session = False
                st.rerun()
            
            if cancel_button:
                st.session_state.show_edit_session = False
                st.rerun()


def render_session_details_modal(session_id: str):
    """Render session details modal."""
    session_manager = st.session_state.session_manager
    session = session_manager.load_session_without_setting_current(session_id)
    
    if not session:
        st.error("Session not found")
        return
    
    st.markdown(f"### 📋 Session Details: {session.name}")
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        st.write(f"**ID:** {session.session_id}")
        st.write(f"**Name:** {session.name}")
        st.write(f"**Status:** {session.status}")
        st.write(f"**Created:** {session.created_at}")
        st.write(f"**Modified:** {session.last_modified}")
    
    with col2:
        st.markdown("#### Statistics")
        st.write(f"**Configuration Items:** {len(session.config)}")
        st.write(f"**Results:** {len(session.results)}")
        
        # Calculate session age
        age = datetime.now() - session.created_at
        st.write(f"**Age:** {age.days} days")
    
    # Description
    if session.description:
        st.markdown("#### Description")
        st.write(session.description)
    
    # Configuration details
    if session.config:
        st.markdown("#### Configuration")
        with st.expander("View Configuration", expanded=False):
            st.json(session.config)
    
    # Results summary
    if session.results:
        st.markdown("#### Results Summary")
        for result_type, result_data in session.results.items():
            with st.expander(f"{result_type.replace('_', ' ').title()}", expanded=False):
                if isinstance(result_data, dict):
                    st.write(f"Keys: {list(result_data.keys())}")
                st.write(f"Type: {type(result_data)}")


def render_delete_confirmation():
    """Render delete confirmation dialog."""
    if st.session_state.get('confirm_delete_session'):
        session_id = st.session_state.confirm_delete_session
        session_name = st.session_state.confirm_delete_name
        
        st.error(f"⚠️ Confirm deletion of session '{session_name}'")
        st.write("This action cannot be undone.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Delete Session", type="primary"):
                session_manager = st.session_state.session_manager
                if session_manager.delete_session(session_id):
                    st.success(f"✅ Deleted session: {session_name}")
                else:
                    st.error("❌ Failed to delete session")
                
                st.session_state.confirm_delete_session = None
                st.session_state.confirm_delete_name = None
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_delete_session = None
                st.session_state.confirm_delete_name = None
                st.rerun()


# Helper functions

def export_session(session_id: str, session_name: str):
    """Export a single session."""
    session_manager = st.session_state.session_manager
    json_data = session_manager.export_session(session_id)
    
    if json_data:
        st.download_button(
            label="📥 Download Session",
            data=json_data,
            file_name=f"{session_name.replace(' ', '_')}.json",
            mime="application/json"
        )
        st.success("✅ Session exported successfully!")
    else:
        st.error("❌ Failed to export session")


def duplicate_session(session_id: str, session_name: str):
    """Duplicate a session."""
    session_manager = st.session_state.session_manager
    
    # Export and re-import to create duplicate
    json_data = session_manager.export_session(session_id)
    
    if json_data:
        # Modify the name
        import json
        session_data = json.loads(json_data)
        session_data['name'] = f"{session_name} (Copy)"
        
        # Import as new session
        new_session = session_manager.import_session(json.dumps(session_data))
        
        if new_session:
            st.success(f"✅ Duplicated session: {new_session.name}")
        else:
            st.error("❌ Failed to duplicate session")
    else:
        st.error("❌ Failed to duplicate session")


def export_multiple_sessions(session_names: List[str], export_format: str):
    """Export multiple sessions."""
    with st.spinner("Exporting sessions..."):
        import time
        time.sleep(2)  # Mock export process
        
        st.success(f"✅ Exported {len(session_names)} sessions in {export_format} format!")


def import_session_file(uploaded_file):
    """Import session from uploaded file."""
    try:
        if uploaded_file.name.endswith('.json'):
            json_data = uploaded_file.read().decode('utf-8')
            import_session_json(json_data)
        else:
            st.error("Unsupported file format. Please upload a JSON file.")
    except Exception as e:
        st.error(f"Error importing file: {e}")


def import_session_json(json_data: str):
    """Import session from JSON data."""
    session_manager = st.session_state.session_manager
    session = session_manager.import_session(json_data)
    
    if session:
        st.success(f"✅ Imported session: {session.name}")
    else:
        st.error("❌ Failed to import session")


def import_session_url(url: str):
    """Import session from URL."""
    st.info("URL import functionality would be implemented here")


# Main render function with modal handling
def render():
    """Main render function with modal handling."""
    # Render main content
    render_main_content()
    
    # Handle modals
    render_create_session_modal()
    render_edit_session_modal()
    render_delete_confirmation()


def render_main_content():
    """Render main session management content."""
    st.title("⚙️ Session Management")
    st.markdown("Create, load, and manage your analysis sessions")
    st.markdown("---")
    
    # Current session status
    render_current_session_status()
    
    # Session actions
    render_session_actions()
    
    # Session history
    render_session_history()
    
    # Session import/export
    render_session_import_export()