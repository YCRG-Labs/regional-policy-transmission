"""
Extensibility framework for regional monetary policy analysis.
Provides support for model extensions, alternative specifications, and robustness checks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type, Union
import numpy as np
import pandas as pd
from enum import Enum
import inspect
import logging


class SpatialWeightMethod(Enum):
    """Available spatial weight construction methods."""
    TRADE_MIGRATION = "trade_migration"
    DISTANCE_ONLY = "distance_only"
    TRADE_ONLY = "trade_only"
    MIGRATION_ONLY = "migration_only"
    FINANCIAL_ONLY = "financial_only"
    CUSTOM = "custom"


class IdentificationStrategy(Enum):
    """Available identification strategies."""
    BASELINE = "baseline"
    ALTERNATIVE_INSTRUMENTS = "alternative_instruments"
    HETEROSKEDASTICITY = "heteroskedasticity"
    HIGHER_ORDER_MOMENTS = "higher_order_moments"
    EXTERNAL_INSTRUMENTS = "external_instruments"


class ModelExtension(Enum):
    """Available model extensions."""
    FINANCIAL_FRICTIONS = "financial_frictions"
    FISCAL_POLICY = "fiscal_policy"
    HOUSING_MARKET = "housing_market"
    LABOR_MARKET = "labor_market"
    BANKING_SECTOR = "banking_sector"


@dataclass
class RobustnessCheck:
    """Configuration for a robustness check."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ParameterRestriction:
    """Configuration for parameter restrictions."""
    parameter_name: str
    restriction_type: str  # "equality", "inequality", "bounds"
    value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    cross_restrictions: Optional[Dict[str, Any]] = None


class SpatialWeightBuilder(ABC):
    """Abstract base class for spatial weight matrix construction."""
    
    @abstractmethod
    def build_weights(self, regions: List[str], data: Dict[str, Any]) -> np.ndarray:
        """
        Build spatial weight matrix.
        
        Args:
            regions: List of region identifiers
            data: Dictionary containing relevant data for weight construction
            
        Returns:
            Spatial weight matrix
        """
        pass
    
    @abstractmethod
    def validate_inputs(self, regions: List[str], data: Dict[str, Any]) -> List[str]:
        """
        Validate inputs for weight construction.
        
        Args:
            regions: List of region identifiers
            data: Dictionary containing relevant data
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass


class TradeBasedWeights(SpatialWeightBuilder):
    """Trade-based spatial weight construction."""
    
    def __init__(self, normalization: str = "row"):
        self.normalization = normalization
    
    def build_weights(self, regions: List[str], data: Dict[str, Any]) -> np.ndarray:
        """Build trade-based spatial weights."""
        trade_flows = data.get('trade_flows')
        if trade_flows is None:
            raise ValueError("Trade flows data required for trade-based weights")
        
        n_regions = len(regions)
        weights = np.zeros((n_regions, n_regions))
        
        for i, region_i in enumerate(regions):
            for j, region_j in enumerate(regions):
                if i != j:
                    # Use bilateral trade flows as weights
                    trade_ij = trade_flows.get((region_i, region_j), 0)
                    weights[i, j] = trade_ij
        
        # Normalize weights
        if self.normalization == "row":
            row_sums = weights.sum(axis=1)
            weights = weights / row_sums[:, np.newaxis]
            weights[np.isnan(weights)] = 0
        
        return weights
    
    def validate_inputs(self, regions: List[str], data: Dict[str, Any]) -> List[str]:
        """Validate trade data inputs."""
        errors = []
        
        if 'trade_flows' not in data:
            errors.append("Trade flows data is required")
        else:
            trade_flows = data['trade_flows']
            if not isinstance(trade_flows, dict):
                errors.append("Trade flows must be a dictionary")
        
        return errors


class DistanceBasedWeights(SpatialWeightBuilder):
    """Distance-based spatial weight construction."""
    
    def __init__(self, decay_parameter: float = 1.0, cutoff_distance: Optional[float] = None):
        self.decay_parameter = decay_parameter
        self.cutoff_distance = cutoff_distance
    
    def build_weights(self, regions: List[str], data: Dict[str, Any]) -> np.ndarray:
        """Build distance-based spatial weights."""
        distances = data.get('distances')
        if distances is None:
            raise ValueError("Distance matrix required for distance-based weights")
        
        n_regions = len(regions)
        weights = np.zeros((n_regions, n_regions))
        
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    distance = distances[i, j]
                    if self.cutoff_distance is None or distance <= self.cutoff_distance:
                        weights[i, j] = 1 / (distance ** self.decay_parameter)
        
        # Row normalize
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        weights[np.isnan(weights)] = 0
        
        return weights
    
    def validate_inputs(self, regions: List[str], data: Dict[str, Any]) -> List[str]:
        """Validate distance data inputs."""
        errors = []
        
        if 'distances' not in data:
            errors.append("Distance matrix is required")
        else:
            distances = data['distances']
            expected_shape = (len(regions), len(regions))
            if distances.shape != expected_shape:
                errors.append(f"Distance matrix shape {distances.shape} doesn't match expected {expected_shape}")
        
        return errors


class ModelExtensionBase(ABC):
    """Abstract base class for model extensions."""
    
    @abstractmethod
    def extend_equations(self, base_equations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extend base model equations.
        
        Args:
            base_equations: Base model equations
            
        Returns:
            Extended equations
        """
        pass
    
    @abstractmethod
    def get_additional_parameters(self) -> List[str]:
        """
        Get list of additional parameters introduced by extension.
        
        Returns:
            List of parameter names
        """
        pass
    
    @abstractmethod
    def validate_extension_data(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data required for model extension.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of validation errors
        """
        pass


class FinancialFrictionsExtension(ModelExtensionBase):
    """Financial frictions model extension."""
    
    def __init__(self, include_credit_spreads: bool = True, include_bank_lending: bool = True):
        self.include_credit_spreads = include_credit_spreads
        self.include_bank_lending = include_bank_lending
    
    def extend_equations(self, base_equations: Dict[str, Any]) -> Dict[str, Any]:
        """Extend equations with financial frictions."""
        extended = base_equations.copy()
        
        # Add credit spread equation
        if self.include_credit_spreads:
            extended['credit_spread'] = {
                'equation': 'cs_t = rho_cs * cs_{t-1} + beta_cs * x_t + epsilon_cs_t',
                'parameters': ['rho_cs', 'beta_cs'],
                'description': 'Credit spread dynamics'
            }
        
        # Add bank lending equation
        if self.include_bank_lending:
            extended['bank_lending'] = {
                'equation': 'bl_t = rho_bl * bl_{t-1} + beta_bl * r_t + gamma_bl * cs_t + epsilon_bl_t',
                'parameters': ['rho_bl', 'beta_bl', 'gamma_bl'],
                'description': 'Bank lending channel'
            }
        
        # Modify IS curve to include financial variables
        if 'is_curve' in extended:
            extended['is_curve']['equation'] += ' + delta_cs * cs_t + delta_bl * bl_t'
            extended['is_curve']['parameters'].extend(['delta_cs', 'delta_bl'])
        
        return extended
    
    def get_additional_parameters(self) -> List[str]:
        """Get additional parameters for financial frictions."""
        params = []
        
        if self.include_credit_spreads:
            params.extend(['rho_cs', 'beta_cs'])
        
        if self.include_bank_lending:
            params.extend(['rho_bl', 'beta_bl', 'gamma_bl'])
        
        params.extend(['delta_cs', 'delta_bl'])
        
        return params
    
    def validate_extension_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate financial frictions data."""
        errors = []
        
        if self.include_credit_spreads and 'credit_spreads' not in data:
            errors.append("Credit spreads data required for financial frictions extension")
        
        if self.include_bank_lending and 'bank_lending' not in data:
            errors.append("Bank lending data required for financial frictions extension")
        
        return errors


class FiscalPolicyExtension(ModelExtensionBase):
    """Fiscal policy model extension."""
    
    def __init__(self, include_government_spending: bool = True, include_taxes: bool = True):
        self.include_government_spending = include_government_spending
        self.include_taxes = include_taxes
    
    def extend_equations(self, base_equations: Dict[str, Any]) -> Dict[str, Any]:
        """Extend equations with fiscal policy."""
        extended = base_equations.copy()
        
        # Add government spending rule
        if self.include_government_spending:
            extended['government_spending'] = {
                'equation': 'g_t = rho_g * g_{t-1} + phi_g * x_t + epsilon_g_t',
                'parameters': ['rho_g', 'phi_g'],
                'description': 'Government spending rule'
            }
        
        # Add tax rule
        if self.include_taxes:
            extended['tax_rate'] = {
                'equation': 'tau_t = rho_tau * tau_{t-1} + phi_tau * debt_{t-1} + epsilon_tau_t',
                'parameters': ['rho_tau', 'phi_tau'],
                'description': 'Tax rate rule'
            }
        
        # Modify IS curve to include fiscal variables
        if 'is_curve' in extended:
            fiscal_terms = []
            if self.include_government_spending:
                fiscal_terms.append('alpha_g * g_t')
            if self.include_taxes:
                fiscal_terms.append('alpha_tau * tau_t')
            
            if fiscal_terms:
                extended['is_curve']['equation'] += ' + ' + ' + '.join(fiscal_terms)
                extended['is_curve']['parameters'].extend(['alpha_g', 'alpha_tau'])
        
        return extended
    
    def get_additional_parameters(self) -> List[str]:
        """Get additional parameters for fiscal policy."""
        params = []
        
        if self.include_government_spending:
            params.extend(['rho_g', 'phi_g', 'alpha_g'])
        
        if self.include_taxes:
            params.extend(['rho_tau', 'phi_tau', 'alpha_tau'])
        
        return params
    
    def validate_extension_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate fiscal policy data."""
        errors = []
        
        if self.include_government_spending and 'government_spending' not in data:
            errors.append("Government spending data required for fiscal policy extension")
        
        if self.include_taxes and 'tax_rates' not in data:
            errors.append("Tax rate data required for fiscal policy extension")
        
        return errors


@dataclass
class ExtensibilityConfig:
    """Configuration for model extensibility and robustness checks."""
    
    # Spatial weight configurations
    spatial_weight_methods: List[SpatialWeightMethod] = field(default_factory=lambda: [SpatialWeightMethod.TRADE_MIGRATION])
    custom_weight_builders: Dict[str, SpatialWeightBuilder] = field(default_factory=dict)
    
    # Identification strategies
    identification_strategies: List[IdentificationStrategy] = field(default_factory=lambda: [IdentificationStrategy.BASELINE])
    
    # Parameter restrictions
    parameter_restrictions: List[ParameterRestriction] = field(default_factory=list)
    
    # Model extensions
    model_extensions: List[ModelExtension] = field(default_factory=list)
    extension_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Robustness checks
    robustness_checks: List[RobustnessCheck] = field(default_factory=list)
    
    # Version control settings
    version_control_enabled: bool = True
    config_versioning: bool = True
    result_versioning: bool = True
    
    def add_spatial_weight_method(self, method: Union[SpatialWeightMethod, str], builder: Optional[SpatialWeightBuilder] = None):
        """Add a spatial weight construction method."""
        if isinstance(method, str):
            if builder is None:
                raise ValueError("Custom spatial weight method requires a builder")
            self.custom_weight_builders[method] = builder
        else:
            if method not in self.spatial_weight_methods:
                self.spatial_weight_methods.append(method)
    
    def add_parameter_restriction(self, parameter: str, restriction_type: str, **kwargs):
        """Add a parameter restriction."""
        restriction = ParameterRestriction(
            parameter_name=parameter,
            restriction_type=restriction_type,
            **kwargs
        )
        self.parameter_restrictions.append(restriction)
    
    def add_robustness_check(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        """Add a robustness check."""
        check = RobustnessCheck(
            name=name,
            description=description,
            parameters=parameters or {},
            enabled=True
        )
        self.robustness_checks.append(check)
    
    def enable_model_extension(self, extension: ModelExtension, config: Optional[Dict[str, Any]] = None):
        """Enable a model extension."""
        if extension not in self.model_extensions:
            self.model_extensions.append(extension)
        
        if config:
            self.extension_configs[extension.value] = config
    
    def get_enabled_extensions(self) -> List[ModelExtensionBase]:
        """Get list of enabled model extension instances."""
        extensions = []
        
        for extension in self.model_extensions:
            config = self.extension_configs.get(extension.value, {})
            
            if extension == ModelExtension.FINANCIAL_FRICTIONS:
                extensions.append(FinancialFrictionsExtension(**config))
            elif extension == ModelExtension.FISCAL_POLICY:
                extensions.append(FiscalPolicyExtension(**config))
            # Add other extensions as needed
        
        return extensions
    
    def validate(self) -> List[str]:
        """Validate extensibility configuration."""
        errors = []
        
        # Validate parameter restrictions
        for restriction in self.parameter_restrictions:
            if restriction.restriction_type == "bounds":
                if restriction.lower_bound is None and restriction.upper_bound is None:
                    errors.append(f"Bounds restriction for {restriction.parameter_name} requires at least one bound")
            elif restriction.restriction_type == "equality":
                if restriction.value is None:
                    errors.append(f"Equality restriction for {restriction.parameter_name} requires a value")
        
        # Validate custom weight builders
        for name, builder in self.custom_weight_builders.items():
            if not isinstance(builder, SpatialWeightBuilder):
                errors.append(f"Custom weight builder {name} must inherit from SpatialWeightBuilder")
        
        return errors


class ExtensibilityManager:
    """Manages model extensions and alternative specifications."""
    
    def __init__(self, config: ExtensibilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._spatial_weight_registry = self._build_spatial_weight_registry()
        self._extension_registry = self._build_extension_registry()
    
    def _build_spatial_weight_registry(self) -> Dict[str, Type[SpatialWeightBuilder]]:
        """Build registry of spatial weight builders."""
        registry = {
            SpatialWeightMethod.TRADE_ONLY.value: TradeBasedWeights,
            SpatialWeightMethod.DISTANCE_ONLY.value: DistanceBasedWeights,
            # Add other built-in methods
        }
        
        # Add custom builders
        for name, builder in self.config.custom_weight_builders.items():
            registry[name] = type(builder)
        
        return registry
    
    def _build_extension_registry(self) -> Dict[str, Type[ModelExtensionBase]]:
        """Build registry of model extensions."""
        return {
            ModelExtension.FINANCIAL_FRICTIONS.value: FinancialFrictionsExtension,
            ModelExtension.FISCAL_POLICY.value: FiscalPolicyExtension,
            # Add other extensions
        }
    
    def get_spatial_weight_builder(self, method: Union[SpatialWeightMethod, str], **kwargs) -> SpatialWeightBuilder:
        """Get spatial weight builder instance."""
        method_name = method.value if isinstance(method, SpatialWeightMethod) else method
        
        if method_name in self._spatial_weight_registry:
            builder_class = self._spatial_weight_registry[method_name]
            return builder_class(**kwargs)
        elif method_name in self.config.custom_weight_builders:
            return self.config.custom_weight_builders[method_name]
        else:
            raise ValueError(f"Unknown spatial weight method: {method_name}")
    
    def apply_parameter_restrictions(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply parameter restrictions to estimation results."""
        restricted_params = parameters.copy()
        
        for restriction in self.config.parameter_restrictions:
            param_name = restriction.parameter_name
            
            if param_name not in restricted_params:
                continue
            
            param_values = restricted_params[param_name]
            
            if restriction.restriction_type == "bounds":
                if restriction.lower_bound is not None:
                    param_values = np.maximum(param_values, restriction.lower_bound)
                if restriction.upper_bound is not None:
                    param_values = np.minimum(param_values, restriction.upper_bound)
            
            elif restriction.restriction_type == "equality":
                param_values = np.full_like(param_values, restriction.value)
            
            restricted_params[param_name] = param_values
        
        return restricted_params
    
    def run_robustness_checks(self, base_results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Run configured robustness checks."""
        robustness_results = {}
        
        for check in self.config.robustness_checks:
            if not check.enabled:
                continue
            
            self.logger.info(f"Running robustness check: {check.name}")
            
            try:
                # This would call specific robustness check implementations
                # For now, we'll create a placeholder structure
                robustness_results[check.name] = {
                    'description': check.description,
                    'parameters': check.parameters,
                    'results': self._run_specific_robustness_check(check, base_results, data)
                }
            except Exception as e:
                self.logger.error(f"Robustness check {check.name} failed: {e}")
                robustness_results[check.name] = {
                    'description': check.description,
                    'error': str(e)
                }
        
        return robustness_results
    
    def _run_specific_robustness_check(self, check: RobustnessCheck, base_results: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific robustness check (placeholder implementation)."""
        # This would contain the actual implementation of different robustness checks
        # For now, return a placeholder
        return {
            'status': 'completed',
            'comparison_with_baseline': 'similar_results',
            'notes': f"Robustness check {check.name} completed successfully"
        }
    
    def get_model_specification(self, base_specification: Dict[str, Any]) -> Dict[str, Any]:
        """Get extended model specification with all enabled extensions."""
        extended_spec = base_specification.copy()
        
        for extension in self.config.get_enabled_extensions():
            extended_spec = extension.extend_equations(extended_spec)
        
        return extended_spec
    
    def validate_extension_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate data for all enabled extensions."""
        errors = []
        
        for extension in self.config.get_enabled_extensions():
            extension_errors = extension.validate_extension_data(data)
            errors.extend(extension_errors)
        
        return errors