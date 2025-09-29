"""
Configuration settings classes for regional monetary policy analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class DataSettings:
    """Configuration for data acquisition and processing."""
    
    # FRED API settings
    fred_api_key: str
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    
    # Regional data configuration
    regions: List[str] = field(default_factory=lambda: [
        "US-Northeast", "US-Southeast", "US-Midwest", "US-Southwest", "US-West"
    ])
    
    # Data series configuration
    output_gap_series: Dict[str, str] = field(default_factory=dict)
    inflation_series: Dict[str, str] = field(default_factory=dict)
    interest_rate_series: str = "FEDFUNDS"
    
    # Time period settings
    start_date: str = "2000-01-01"
    end_date: str = "2023-12-31"
    frequency: str = "monthly"  # monthly, quarterly
    
    # Data processing options
    real_time_data: bool = True
    vintage_tracking: bool = True
    data_transformations: List[str] = field(default_factory=list)
    
    # Caching settings
    cache_enabled: bool = True
    cache_directory: str = "data/cache"
    cache_expiry_days: int = 7
    
    def __post_init__(self):
        """Set default series mappings if not provided."""
        if not self.output_gap_series:
            self.output_gap_series = {
                region: f"{region}_OUTPUT_GAP" for region in self.regions
            }
        
        if not self.inflation_series:
            self.inflation_series = {
                region: f"{region}_INFLATION" for region in self.regions
            }


@dataclass
class EstimationSettings:
    """Configuration for econometric estimation procedures."""
    
    # Estimation method
    estimation_method: str = "three_stage_gmm"
    identification_strategy: str = "baseline"
    
    # Spatial weight construction
    spatial_weight_method: str = "trade_migration"
    spatial_weight_params: Dict[str, float] = field(default_factory=lambda: {
        "trade_weight": 0.4,
        "migration_weight": 0.3,
        "financial_weight": 0.2,
        "distance_weight": 0.1
    })
    
    # GMM estimation options
    gmm_options: Dict[str, Any] = field(default_factory=lambda: {
        "weighting_matrix": "optimal",
        "moment_conditions": "standard",
        "instrument_selection": "automatic"
    })
    
    # Numerical optimization
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    optimization_method: str = "BFGS"
    
    # Standard error computation
    standard_error_method: str = "bootstrap"
    bootstrap_replications: int = 1000
    bootstrap_block_size: int = 12  # For time series bootstrap
    confidence_level: float = 0.95
    
    # Robustness checks
    robustness_checks: List[str] = field(default_factory=lambda: [
        "alternative_instruments",
        "subsample_stability", 
        "parameter_restrictions"
    ])
    
    # Identification tests
    identification_tests: List[str] = field(default_factory=lambda: [
        "weak_identification",
        "overidentification",
        "parameter_stability"
    ])


@dataclass
class PolicySettings:
    """Configuration for policy analysis and counterfactuals."""
    
    # Policy mistake decomposition
    decomposition_method: str = "theorem_4"
    information_set_reconstruction: str = "real_time"
    
    # Counterfactual scenarios
    scenarios_to_run: List[str] = field(default_factory=lambda: [
        "baseline", "perfect_info", "optimal_regional", "perfect_regional"
    ])
    
    # Welfare function parameters
    welfare_function: str = "quadratic_loss"
    social_weights: Optional[List[float]] = None  # Equal weights if None
    discount_factor: float = 0.99
    
    # Policy rule parameters
    taylor_rule_estimation: bool = True
    taylor_rule_params: Dict[str, float] = field(default_factory=lambda: {
        "inflation_coefficient": 1.5,
        "output_gap_coefficient": 0.5,
        "interest_rate_smoothing": 0.8
    })
    
    # Counterfactual analysis options
    parallel_processing: bool = True
    n_simulation_draws: int = 1000
    confidence_bands: bool = True


@dataclass
class VisualizationSettings:
    """Configuration for visualization and reporting."""
    
    # Plot settings
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "Set2"
    
    # Regional map settings
    map_projection: str = "albers_usa"
    choropleth_colorscale: str = "RdYlBu"
    
    # Time series plot settings
    time_series_style: str = "line"
    confidence_interval_alpha: float = 0.3
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "svg"])
    export_directory: str = "output/figures"
    
    # Report generation
    report_template: str = "academic"
    include_methodology: bool = True
    include_robustness_checks: bool = True


@dataclass
class AnalysisSettings:
    """Master configuration combining all analysis settings."""
    
    data: DataSettings
    estimation: EstimationSettings
    policy: PolicySettings
    visualization: VisualizationSettings
    
    # General analysis options
    analysis_name: str = "regional_monetary_policy_analysis"
    output_directory: str = "output"
    log_level: str = "INFO"
    random_seed: int = 42
    
    # Performance settings
    n_cores: int = -1  # Use all available cores
    memory_limit_gb: Optional[float] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AnalysisSettings':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            AnalysisSettings instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisSettings':
        """
        Create AnalysisSettings from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            AnalysisSettings instance
        """
        data_settings = DataSettings(**config_dict.get('data', {}))
        estimation_settings = EstimationSettings(**config_dict.get('estimation', {}))
        policy_settings = PolicySettings(**config_dict.get('policy', {}))
        visualization_settings = VisualizationSettings(**config_dict.get('visualization', {}))
        
        # Extract general settings
        general_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['data', 'estimation', 'policy', 'visualization']}
        
        return cls(
            data=data_settings,
            estimation=estimation_settings,
            policy=policy_settings,
            visualization=visualization_settings,
            **general_settings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary format.
        
        Returns:
            Dictionary representation of settings
        """
        return {
            'data': self.data.__dict__,
            'estimation': self.estimation.__dict__,
            'policy': self.policy.__dict__,
            'visualization': self.visualization.__dict__,
            'analysis_name': self.analysis_name,
            'output_directory': self.output_directory,
            'log_level': self.log_level,
            'random_seed': self.random_seed,
            'n_cores': self.n_cores,
            'memory_limit_gb': self.memory_limit_gb
        }
    
    def save_to_file(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path where to save configuration
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate data settings
        if not self.data.fred_api_key:
            errors.append("FRED API key is required")
        
        if not self.data.regions:
            errors.append("At least one region must be specified")
        
        # Validate estimation settings
        if self.estimation.convergence_tolerance <= 0:
            errors.append("Convergence tolerance must be positive")
        
        if self.estimation.max_iterations <= 0:
            errors.append("Max iterations must be positive")
        
        # Validate spatial weights sum to 1
        weight_sum = sum(self.estimation.spatial_weight_params.values())
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append("Spatial weight parameters must sum to 1")
        
        # Validate policy settings
        if self.policy.social_weights and len(self.policy.social_weights) != len(self.data.regions):
            errors.append("Social weights must match number of regions")
        
        return errors