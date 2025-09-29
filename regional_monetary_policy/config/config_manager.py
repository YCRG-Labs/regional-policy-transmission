"""
Configuration manager for regional monetary policy analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .settings import AnalysisSettings, DataSettings, EstimationSettings, PolicySettings, VisualizationSettings
from .extensibility import ExtensibilityConfig, ExtensibilityManager
from .version_control import VersionControlManager


class ConfigManager:
    """
    Manages configuration loading, validation, and environment setup.
    """
    
    def __init__(self, config_path: Optional[str] = None, enable_version_control: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
            enable_version_control: Whether to enable version control
        """
        self.config_path = config_path
        self.settings: Optional[AnalysisSettings] = None
        self.extensibility_config: Optional[ExtensibilityConfig] = None
        self.extensibility_manager: Optional[ExtensibilityManager] = None
        self.version_control: Optional[VersionControlManager] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize version control if enabled
        if enable_version_control:
            try:
                self.version_control = VersionControlManager()
            except Exception as e:
                self.logger.warning(f"Could not initialize version control: {e}")
                self.version_control = None
    
    def load_config(self, config_path: Optional[str] = None) -> AnalysisSettings:
        """
        Load configuration from file or create default configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded AnalysisSettings
        """
        if config_path:
            self.config_path = config_path
        
        if self.config_path and Path(self.config_path).exists():
            self.logger.info(f"Loading configuration from {self.config_path}")
            self.settings = AnalysisSettings.from_file(self.config_path)
        else:
            self.logger.info("Creating default configuration")
            self.settings = self._create_default_config()
        
        # Validate configuration
        validation_errors = self.settings.validate()
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        
        # Setup environment based on configuration
        self._setup_environment()
        
        return self.settings
    
    def _create_default_config(self) -> AnalysisSettings:
        """
        Create default configuration settings.
        
        Returns:
            Default AnalysisSettings
        """
        # Get FRED API key from environment variable
        fred_api_key = os.getenv('FRED_API_KEY', '')
        if not fred_api_key:
            self.logger.warning("FRED_API_KEY environment variable not set")
        
        data_settings = DataSettings(fred_api_key=fred_api_key)
        estimation_settings = EstimationSettings()
        policy_settings = PolicySettings()
        visualization_settings = VisualizationSettings()
        
        # Create default extensibility configuration
        self.extensibility_config = ExtensibilityConfig()
        
        return AnalysisSettings(
            data=data_settings,
            estimation=estimation_settings,
            policy=policy_settings,
            visualization=visualization_settings
        )
    
    def _setup_environment(self):
        """Setup environment based on configuration settings."""
        if not self.settings:
            return
        
        # Create output directories
        output_dir = Path(self.settings.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "figures").mkdir(exist_ok=True)
        (output_dir / "tables").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
        
        # Create cache directory
        cache_dir = Path(self.settings.data.cache_directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seed for reproducibility
        import numpy as np
        np.random.seed(self.settings.random_seed)
        
        # Initialize extensibility manager
        if self.extensibility_config:
            self.extensibility_manager = ExtensibilityManager(self.extensibility_config)
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        if not self.settings:
            return
        
        log_level = getattr(logging, self.settings.log_level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        log_file = Path(self.settings.output_directory) / "analysis.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def save_config(self, config_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path where to save configuration
        """
        if not self.settings:
            raise ValueError("No configuration loaded to save")
        
        save_path = config_path or self.config_path or "config.json"
        self.settings.save_to_file(save_path)
        self.logger.info(f"Configuration saved to {save_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if not self.settings:
            raise ValueError("No configuration loaded to update")
        
        # Convert current settings to dict, update, and reload
        current_config = self.settings.to_dict()
        current_config.update(updates)
        
        self.settings = AnalysisSettings.from_dict(current_config)
        
        # Validate updated configuration
        validation_errors = self.settings.validate()
        if validation_errors:
            error_msg = "Updated configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
    
    def get_setting(self, setting_path: str) -> Any:
        """
        Get a specific setting using dot notation.
        
        Args:
            setting_path: Dot-separated path to setting (e.g., 'data.fred_api_key')
            
        Returns:
            Setting value
        """
        if not self.settings:
            raise ValueError("No configuration loaded")
        
        parts = setting_path.split('.')
        current = self.settings
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise KeyError(f"Setting '{setting_path}' not found")
        
        return current
    
    def set_setting(self, setting_path: str, value: Any):
        """
        Set a specific setting using dot notation.
        
        Args:
            setting_path: Dot-separated path to setting
            value: New value for the setting
        """
        if not self.settings:
            raise ValueError("No configuration loaded")
        
        parts = setting_path.split('.')
        current = self.settings
        
        # Navigate to parent of target setting
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise KeyError(f"Setting path '{'.'.join(parts[:-1])}' not found")
        
        # Set the final setting
        final_part = parts[-1]
        if hasattr(current, final_part):
            setattr(current, final_part, value)
        else:
            raise KeyError(f"Setting '{setting_path}' not found")
    
    def create_config_template(self, output_path: str):
        """
        Create a configuration template file with all available options.
        
        Args:
            output_path: Path where to save the template
        """
        template_config = self._create_default_config()
        
        # Add comments to the template
        config_dict = template_config.to_dict()
        
        # Add documentation
        config_dict['_documentation'] = {
            'description': 'Regional Monetary Policy Analysis Configuration',
            'sections': {
                'data': 'Data acquisition and processing settings',
                'estimation': 'Econometric estimation configuration',
                'policy': 'Policy analysis and counterfactual settings',
                'visualization': 'Plotting and reporting options'
            },
            'required_env_vars': ['FRED_API_KEY'],
            'example_usage': 'config_manager.load_config("config.json")'
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration template created at {output_path}")
    
    @property
    def is_configured(self) -> bool:
        """Check if configuration is loaded and valid."""
        return self.settings is not None
    
    def load_extensibility_config(self, extensibility_config: Optional[ExtensibilityConfig] = None):
        """
        Load extensibility configuration.
        
        Args:
            extensibility_config: Extensibility configuration to use
        """
        if extensibility_config:
            self.extensibility_config = extensibility_config
        elif self.extensibility_config is None:
            self.extensibility_config = ExtensibilityConfig()
        
        # Validate extensibility configuration
        validation_errors = self.extensibility_config.validate()
        if validation_errors:
            error_msg = "Extensibility configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
        
        # Initialize extensibility manager
        self.extensibility_manager = ExtensibilityManager(self.extensibility_config)
        self.logger.info("Loaded extensibility configuration")
    
    def save_config_version(self, description: str = "", tags: Optional[List[str]] = None) -> Optional[str]:
        """
        Save current configuration as a new version.
        
        Args:
            description: Description of the configuration version
            tags: Tags to associate with the version
            
        Returns:
            Version ID if version control is enabled, None otherwise
        """
        if not self.version_control or not self.settings:
            return None
        
        # Combine main settings and extensibility config
        config_data = self.settings.to_dict()
        if self.extensibility_config:
            config_data['extensibility'] = {
                'spatial_weight_methods': [method.value for method in self.extensibility_config.spatial_weight_methods],
                'identification_strategies': [strategy.value for strategy in self.extensibility_config.identification_strategies],
                'parameter_restrictions': [
                    {
                        'parameter_name': r.parameter_name,
                        'restriction_type': r.restriction_type,
                        'value': r.value,
                        'lower_bound': r.lower_bound,
                        'upper_bound': r.upper_bound
                    }
                    for r in self.extensibility_config.parameter_restrictions
                ],
                'model_extensions': [ext.value for ext in self.extensibility_config.model_extensions],
                'extension_configs': self.extensibility_config.extension_configs,
                'robustness_checks': [
                    {
                        'name': check.name,
                        'description': check.description,
                        'parameters': check.parameters,
                        'enabled': check.enabled
                    }
                    for check in self.extensibility_config.robustness_checks
                ]
            }
        
        return self.version_control.save_config_version(config_data, description, tags)
    
    def load_config_version(self, version_id: str):
        """
        Load a specific configuration version.
        
        Args:
            version_id: Version ID to load
        """
        if not self.version_control:
            raise ValueError("Version control not enabled")
        
        config_data = self.version_control.load_config_version(version_id)
        
        # Load main settings
        main_config = {k: v for k, v in config_data.items() if k != 'extensibility'}
        self.settings = AnalysisSettings.from_dict(main_config)
        
        # Load extensibility configuration if present
        if 'extensibility' in config_data:
            ext_data = config_data['extensibility']
            
            # Reconstruct extensibility config
            from .extensibility import SpatialWeightMethod, IdentificationStrategy, ModelExtension, RobustnessCheck, ParameterRestriction
            
            self.extensibility_config = ExtensibilityConfig(
                spatial_weight_methods=[SpatialWeightMethod(method) for method in ext_data.get('spatial_weight_methods', [])],
                identification_strategies=[IdentificationStrategy(strategy) for strategy in ext_data.get('identification_strategies', [])],
                model_extensions=[ModelExtension(ext) for ext in ext_data.get('model_extensions', [])],
                extension_configs=ext_data.get('extension_configs', {}),
                parameter_restrictions=[
                    ParameterRestriction(**restriction)
                    for restriction in ext_data.get('parameter_restrictions', [])
                ],
                robustness_checks=[
                    RobustnessCheck(**check)
                    for check in ext_data.get('robustness_checks', [])
                ]
            )
        
        # Setup environment and extensibility manager
        self._setup_environment()
        
        self.logger.info(f"Loaded configuration version {version_id}")
    
    def start_analysis_run(self, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Start a new analysis run with version tracking.
        
        Args:
            parameters: Additional parameters for the run
            
        Returns:
            Run ID if version control is enabled, None otherwise
        """
        if not self.version_control or not self.settings:
            return None
        
        # Combine main settings and extensibility config
        config_data = self.settings.to_dict()
        if self.extensibility_config:
            config_data['extensibility'] = {
                'spatial_weight_methods': [method.value for method in self.extensibility_config.spatial_weight_methods],
                'identification_strategies': [strategy.value for strategy in self.extensibility_config.identification_strategies],
                'parameter_restrictions': [
                    {
                        'parameter_name': r.parameter_name,
                        'restriction_type': r.restriction_type,
                        'value': r.value,
                        'lower_bound': r.lower_bound,
                        'upper_bound': r.upper_bound
                    }
                    for r in self.extensibility_config.parameter_restrictions
                ],
                'model_extensions': [ext.value for ext in self.extensibility_config.model_extensions],
                'extension_configs': self.extensibility_config.extension_configs,
                'robustness_checks': [
                    {
                        'name': check.name,
                        'description': check.description,
                        'parameters': check.parameters,
                        'enabled': check.enabled
                    }
                    for check in self.extensibility_config.robustness_checks
                ]
            }
        
        return self.version_control.start_analysis_run(config_data, parameters)
    
    def complete_analysis_run(self, run_id: str, results_path: Optional[str] = None, 
                            execution_time: Optional[float] = None):
        """
        Complete an analysis run.
        
        Args:
            run_id: Run ID to complete
            results_path: Path to results
            execution_time: Execution time in seconds
        """
        if self.version_control:
            self.version_control.complete_analysis_run(run_id, results_path, execution_time)
    
    def fail_analysis_run(self, run_id: str, error_message: str):
        """
        Mark an analysis run as failed.
        
        Args:
            run_id: Run ID to fail
            error_message: Error message
        """
        if self.version_control:
            self.version_control.fail_analysis_run(run_id, error_message)
    
    def get_spatial_weight_builder(self, method: str, **kwargs):
        """
        Get spatial weight builder for specified method.
        
        Args:
            method: Spatial weight method name
            **kwargs: Additional parameters for the builder
            
        Returns:
            Spatial weight builder instance
        """
        if not self.extensibility_manager:
            raise ValueError("Extensibility manager not initialized")
        
        return self.extensibility_manager.get_spatial_weight_builder(method, **kwargs)
    
    def apply_parameter_restrictions(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configured parameter restrictions.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Restricted parameters
        """
        if not self.extensibility_manager:
            return parameters
        
        return self.extensibility_manager.apply_parameter_restrictions(parameters)
    
    def run_robustness_checks(self, base_results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run configured robustness checks.
        
        Args:
            base_results: Base analysis results
            data: Input data
            
        Returns:
            Robustness check results
        """
        if not self.extensibility_manager:
            return {}
        
        return self.extensibility_manager.run_robustness_checks(base_results, data)
    
    def get_model_specification(self, base_specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get extended model specification with enabled extensions.
        
        Args:
            base_specification: Base model specification
            
        Returns:
            Extended model specification
        """
        if not self.extensibility_manager:
            return base_specification
        
        return self.extensibility_manager.get_model_specification(base_specification)
    
    def validate_extension_data(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data for enabled model extensions.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of validation errors
        """
        if not self.extensibility_manager:
            return []
        
        return self.extensibility_manager.validate_extension_data(data)
    
    def get_config_history(self, limit: Optional[int] = None):
        """
        Get configuration version history.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of configuration versions
        """
        if not self.version_control:
            return []
        
        return self.version_control.get_config_history(limit)
    
    def get_run_history(self, config_version: Optional[str] = None, 
                       status: Optional[str] = None, limit: Optional[int] = None):
        """
        Get analysis run history.
        
        Args:
            config_version: Filter by configuration version
            status: Filter by run status
            limit: Maximum number of runs to return
            
        Returns:
            List of analysis runs
        """
        if not self.version_control:
            return []
        
        return self.version_control.get_run_history(config_version, status, limit)
    
    def create_reproducibility_report(self, run_id: str) -> Dict[str, Any]:
        """
        Create reproducibility report for an analysis run.
        
        Args:
            run_id: Analysis run ID
            
        Returns:
            Reproducibility report
        """
        if not self.version_control:
            raise ValueError("Version control not enabled")
        
        return self.version_control.create_reproducibility_report(run_id)
    
    def export_configuration_template(self, output_path: str, include_extensibility: bool = True):
        """
        Export a comprehensive configuration template.
        
        Args:
            output_path: Path to save template
            include_extensibility: Whether to include extensibility options
        """
        template_config = self._create_default_config()
        config_dict = template_config.to_dict()
        
        # Add extensibility template if requested
        if include_extensibility:
            config_dict['extensibility'] = {
                'spatial_weight_methods': ['trade_migration', 'distance_only'],
                'identification_strategies': ['baseline', 'alternative_instruments'],
                'parameter_restrictions': [
                    {
                        'parameter_name': 'sigma',
                        'restriction_type': 'bounds',
                        'lower_bound': 0.0,
                        'upper_bound': 5.0
                    }
                ],
                'model_extensions': ['financial_frictions'],
                'extension_configs': {
                    'financial_frictions': {
                        'include_credit_spreads': True,
                        'include_bank_lending': True
                    }
                },
                'robustness_checks': [
                    {
                        'name': 'subsample_stability',
                        'description': 'Test parameter stability across subsamples',
                        'parameters': {'split_date': '2008-01-01'},
                        'enabled': True
                    }
                ]
            }
        
        # Add comprehensive documentation
        config_dict['_documentation'] = {
            'description': 'Regional Monetary Policy Analysis Configuration Template',
            'version': '1.0',
            'sections': {
                'data': 'Data acquisition and processing settings',
                'estimation': 'Econometric estimation configuration',
                'policy': 'Policy analysis and counterfactual settings',
                'visualization': 'Plotting and reporting options',
                'extensibility': 'Model extensions and robustness checks'
            },
            'spatial_weight_methods': {
                'trade_migration': 'Combined trade and migration flows',
                'distance_only': 'Geographic distance based',
                'trade_only': 'Trade flows only',
                'custom': 'User-defined weight construction'
            },
            'identification_strategies': {
                'baseline': 'Standard GMM identification',
                'alternative_instruments': 'Alternative instrument set',
                'heteroskedasticity': 'Identification through heteroskedasticity',
                'external_instruments': 'External instrumental variables'
            },
            'model_extensions': {
                'financial_frictions': 'Credit spreads and bank lending channels',
                'fiscal_policy': 'Government spending and tax interactions',
                'housing_market': 'Housing market dynamics',
                'labor_market': 'Labor market frictions'
            },
            'required_env_vars': ['FRED_API_KEY'],
            'example_usage': [
                'config_manager = ConfigManager()',
                'config_manager.load_config("config.json")',
                'run_id = config_manager.start_analysis_run()',
                'config_manager.complete_analysis_run(run_id, "results/")'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration template exported to {output_path}")
    
    @property
    def has_version_control(self) -> bool:
        """Check if version control is available."""
        return self.version_control is not None
    
    @property
    def has_extensibility(self) -> bool:
        """Check if extensibility framework is available."""
        return self.extensibility_manager is not None
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        status = "configured" if self.is_configured else "not configured"
        config_path = self.config_path or "default"
        features = []
        if self.has_version_control:
            features.append("version_control")
        if self.has_extensibility:
            features.append("extensibility")
        
        feature_str = f", features={features}" if features else ""
        return f"ConfigManager(status={status}, config_path={config_path}{feature_str})"