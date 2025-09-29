"""
Web interface module for Regional Monetary Policy Analysis System.

This module provides a Streamlit-based web interface for interactive analysis,
parameter configuration, and result visualization.
"""

from .app import main as run_app
from .session_manager import SessionManager
from .workflow_manager import WorkflowManager

__all__ = ['run_app', 'SessionManager', 'WorkflowManager']