"""
Configuration settings for the web interface.
"""

import os
from typing import Dict, Any

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'Regional Monetary Policy Analysis',
    'page_icon': '🏦',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Default directories
DEFAULT_DIRS = {
    'data': 'data',
    'cache': 'data/cache',
    'sessions': 'data/sessions',
    'exports': 'data/exports',
    'reports': 'data/reports'
}

# Ensure directories exist
for dir_path in DEFAULT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Web interface settings
WEB_CONFIG = {
    'max_file_upload_size': 200,  # MB
    'session_timeout': 3600,  # seconds
    'auto_save_interval': 300,  # seconds
    'max_sessions_per_user': 50,
    'enable_debug_mode': False
}

# Visualization settings
VIZ_CONFIG = {
    'default_color_scheme': 'plotly',
    'chart_height': 400,
    'map_height': 600,
    'dpi': 300,
    'export_formats': ['png', 'pdf', 'svg', 'html']
}

# Analysis settings
ANALYSIS_CONFIG = {
    'max_regions': 50,
    'max_time_periods': 1000,
    'default_confidence_level': 0.95,
    'max_bootstrap_iterations': 10000,
    'parallel_processing': True
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        'streamlit': STREAMLIT_CONFIG,
        'directories': DEFAULT_DIRS,
        'web': WEB_CONFIG,
        'visualization': VIZ_CONFIG,
        'analysis': ANALYSIS_CONFIG
    }