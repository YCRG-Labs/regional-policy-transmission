"""
Session management for the web interface.

Handles user sessions, analysis history, and state persistence.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import streamlit as st
import pandas as pd

from ..config.config_manager import ConfigManager
from ..exceptions import RegionalMonetaryPolicyError


@dataclass
class AnalysisSession:
    """Container for analysis session data."""
    
    session_id: str
    created_at: datetime
    last_modified: datetime
    name: str
    description: str
    config: Dict[str, Any]
    results: Dict[str, Any]
    status: str  # 'created', 'running', 'completed', 'error'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_modified'] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSession':
        """Create session from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


class SessionManager:
    """Manages user sessions and analysis history."""
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        """Initialize session manager.
        
        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
        
        # Initialize session state if not exists
        if 'current_session' not in st.session_state:
            st.session_state.current_session = None
        if 'session_history' not in st.session_state:
            st.session_state.session_history = []
    
    def create_session(self, name: str, description: str = "", 
                      config: Optional[Dict[str, Any]] = None) -> AnalysisSession:
        """Create a new analysis session.
        
        Args:
            name: Session name
            description: Session description
            config: Initial configuration
            
        Returns:
            Created session
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = AnalysisSession(
            session_id=session_id,
            created_at=now,
            last_modified=now,
            name=name,
            description=description,
            config=config or {},
            results={},
            status='created'
        )
        
        # Save session to file
        self._save_session(session)
        
        # Update session state
        st.session_state.current_session = session
        self._update_session_history()
        
        return session
    
    def load_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Load an existing session.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Loaded session or None if not found
        """
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return None
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            session = AnalysisSession.from_dict(data)
            st.session_state.current_session = session
            return session
            
        except Exception as e:
            st.error(f"Error loading session: {e}")
            return None
    
    def save_current_session(self) -> bool:
        """Save the current session.
        
        Returns:
            True if saved successfully
        """
        if st.session_state.current_session is None:
            return False
        
        session = st.session_state.current_session
        session.last_modified = datetime.now()
        
        return self._save_session(session)
    
    def update_session_config(self, config: Dict[str, Any]) -> None:
        """Update current session configuration.
        
        Args:
            config: New configuration
        """
        if st.session_state.current_session is not None:
            st.session_state.current_session.config.update(config)
            st.session_state.current_session.last_modified = datetime.now()
    
    def update_session_results(self, results: Dict[str, Any]) -> None:
        """Update current session results.
        
        Args:
            results: Analysis results
        """
        if st.session_state.current_session is not None:
            st.session_state.current_session.results.update(results)
            st.session_state.current_session.last_modified = datetime.now()
    
    def update_session_status(self, status: str) -> None:
        """Update current session status.
        
        Args:
            status: New status
        """
        if st.session_state.current_session is not None:
            st.session_state.current_session.status = status
            st.session_state.current_session.last_modified = datetime.now()
    
    def get_session_history(self) -> List[AnalysisSession]:
        """Get list of all sessions.
        
        Returns:
            List of sessions sorted by last modified date
        """
        sessions = []
        
        if not os.path.exists(self.sessions_dir):
            return sessions
        
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]  # Remove .json extension
                session = self.load_session_without_setting_current(session_id)
                if session:
                    sessions.append(session)
        
        # Sort by last modified date (newest first)
        sessions.sort(key=lambda x: x.last_modified, reverse=True)
        return sessions
    
    def load_session_without_setting_current(self, session_id: str) -> Optional[AnalysisSession]:
        """Load session without setting it as current.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Loaded session or None if not found
        """
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return None
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            return AnalysisSession.from_dict(data)
            
        except Exception:
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully
        """
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
            
            # Clear current session if it's the one being deleted
            if (st.session_state.current_session and 
                st.session_state.current_session.session_id == session_id):
                st.session_state.current_session = None
            
            self._update_session_history()
            return True
            
        except Exception as e:
            st.error(f"Error deleting session: {e}")
            return False
    
    def export_session(self, session_id: str) -> Optional[str]:
        """Export session to JSON string.
        
        Args:
            session_id: Session ID to export
            
        Returns:
            JSON string or None if error
        """
        session = self.load_session_without_setting_current(session_id)
        if session is None:
            return None
        
        try:
            return json.dumps(session.to_dict(), indent=2)
        except Exception:
            return None
    
    def import_session(self, json_data: str) -> Optional[AnalysisSession]:
        """Import session from JSON string.
        
        Args:
            json_data: JSON string containing session data
            
        Returns:
            Imported session or None if error
        """
        try:
            data = json.loads(json_data)
            
            # Generate new session ID to avoid conflicts
            data['session_id'] = str(uuid.uuid4())
            data['created_at'] = datetime.now().isoformat()
            data['last_modified'] = datetime.now().isoformat()
            
            session = AnalysisSession.from_dict(data)
            self._save_session(session)
            
            return session
            
        except Exception as e:
            st.error(f"Error importing session: {e}")
            return None
    
    def _save_session(self, session: AnalysisSession) -> bool:
        """Save session to file.
        
        Args:
            session: Session to save
            
        Returns:
            True if saved successfully
        """
        session_file = os.path.join(self.sessions_dir, f"{session.session_id}.json")
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
            
        except Exception as e:
            st.error(f"Error saving session: {e}")
            return False
    
    def _update_session_history(self) -> None:
        """Update session history in session state."""
        st.session_state.session_history = self.get_session_history()
    
    def get_current_session(self) -> Optional[AnalysisSession]:
        """Get current session.
        
        Returns:
            Current session or None
        """
        return st.session_state.current_session
    
    def has_current_session(self) -> bool:
        """Check if there's a current session.
        
        Returns:
            True if current session exists
        """
        return st.session_state.current_session is not None