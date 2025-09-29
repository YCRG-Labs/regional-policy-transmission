"""
Workflow management for guided analysis workflows.

Provides structured workflows for different research questions and analysis types.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import streamlit as st

from ..config.config_manager import ConfigManager
from ..exceptions import RegionalMonetaryPolicyError


class WorkflowType(Enum):
    """Types of analysis workflows."""
    PARAMETER_ESTIMATION = "parameter_estimation"
    POLICY_ANALYSIS = "policy_analysis"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    REGIONAL_COMPARISON = "regional_comparison"
    ROBUSTNESS_CHECK = "robustness_check"


@dataclass
class WorkflowStep:
    """Individual step in an analysis workflow."""
    
    step_id: str
    title: str
    description: str
    required_inputs: List[str]
    optional_inputs: List[str]
    validation_func: Optional[Callable] = None
    execution_func: Optional[Callable] = None
    completed: bool = False
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate step inputs.
        
        Args:
            inputs: Input values to validate
            
        Returns:
            True if inputs are valid
        """
        # Check required inputs
        for required in self.required_inputs:
            if required not in inputs or inputs[required] is None:
                return False
        
        # Run custom validation if provided
        if self.validation_func:
            return self.validation_func(inputs)
        
        return True


@dataclass
class AnalysisWorkflow:
    """Complete analysis workflow."""
    
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep]
    current_step: int = 0
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get current workflow step.
        
        Returns:
            Current step or None if workflow complete
        """
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def advance_step(self) -> bool:
        """Advance to next step.
        
        Returns:
            True if advanced successfully
        """
        if self.current_step < len(self.steps):
            self.steps[self.current_step].completed = True
            self.current_step += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if workflow is complete.
        
        Returns:
            True if all steps completed
        """
        return self.current_step >= len(self.steps)
    
    def get_progress(self) -> float:
        """Get workflow progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if not self.steps:
            return 100.0
        return (self.current_step / len(self.steps)) * 100


class WorkflowManager:
    """Manages analysis workflows and guides users through structured analysis."""
    
    def __init__(self):
        """Initialize workflow manager."""
        self.workflows = self._create_workflows()
        
        # Initialize workflow state
        if 'current_workflow' not in st.session_state:
            st.session_state.current_workflow = None
        if 'workflow_data' not in st.session_state:
            st.session_state.workflow_data = {}
    
    def get_available_workflows(self) -> List[AnalysisWorkflow]:
        """Get list of available workflows.
        
        Returns:
            List of available workflows
        """
        return list(self.workflows.values())
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start a new workflow.
        
        Args:
            workflow_id: ID of workflow to start
            
        Returns:
            True if started successfully
        """
        if workflow_id not in self.workflows:
            return False
        
        # Create a copy of the workflow
        workflow = self.workflows[workflow_id]
        new_workflow = AnalysisWorkflow(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            workflow_type=workflow.workflow_type,
            steps=[
                WorkflowStep(
                    step_id=step.step_id,
                    title=step.title,
                    description=step.description,
                    required_inputs=step.required_inputs.copy(),
                    optional_inputs=step.optional_inputs.copy(),
                    validation_func=step.validation_func,
                    execution_func=step.execution_func,
                    completed=False
                ) for step in workflow.steps
            ],
            current_step=0
        )
        
        st.session_state.current_workflow = new_workflow
        st.session_state.workflow_data = {}
        
        return True
    
    def get_current_workflow(self) -> Optional[AnalysisWorkflow]:
        """Get current active workflow.
        
        Returns:
            Current workflow or None
        """
        return st.session_state.current_workflow
    
    def update_workflow_data(self, step_id: str, data: Dict[str, Any]) -> None:
        """Update workflow data for a step.
        
        Args:
            step_id: Step ID
            data: Data to update
        """
        if step_id not in st.session_state.workflow_data:
            st.session_state.workflow_data[step_id] = {}
        
        st.session_state.workflow_data[step_id].update(data)
    
    def get_workflow_data(self, step_id: Optional[str] = None) -> Dict[str, Any]:
        """Get workflow data.
        
        Args:
            step_id: Specific step ID or None for all data
            
        Returns:
            Workflow data
        """
        if step_id:
            return st.session_state.workflow_data.get(step_id, {})
        return st.session_state.workflow_data
    
    def validate_current_step(self) -> bool:
        """Validate current workflow step.
        
        Returns:
            True if current step is valid
        """
        workflow = self.get_current_workflow()
        if not workflow:
            return False
        
        current_step = workflow.get_current_step()
        if not current_step:
            return True  # Workflow complete
        
        step_data = self.get_workflow_data(current_step.step_id)
        return current_step.validate_inputs(step_data)
    
    def advance_workflow(self) -> bool:
        """Advance workflow to next step.
        
        Returns:
            True if advanced successfully
        """
        workflow = self.get_current_workflow()
        if not workflow:
            return False
        
        if not self.validate_current_step():
            return False
        
        return workflow.advance_step()
    
    def reset_workflow(self) -> None:
        """Reset current workflow."""
        st.session_state.current_workflow = None
        st.session_state.workflow_data = {}
    
    def _create_workflows(self) -> Dict[str, AnalysisWorkflow]:
        """Create predefined workflows.
        
        Returns:
            Dictionary of workflows
        """
        workflows = {}
        
        # Parameter Estimation Workflow
        param_steps = [
            WorkflowStep(
                step_id="data_selection",
                title="Data Selection",
                description="Select regions, time period, and economic indicators",
                required_inputs=["regions", "start_date", "end_date", "indicators"],
                optional_inputs=["data_frequency", "vintage_dates"]
            ),
            WorkflowStep(
                step_id="spatial_weights",
                title="Spatial Weight Configuration",
                description="Configure spatial weight matrix construction",
                required_inputs=["weight_components", "weight_values"],
                optional_inputs=["normalization_method", "validation_checks"]
            ),
            WorkflowStep(
                step_id="estimation_setup",
                title="Estimation Setup",
                description="Configure estimation parameters and options",
                required_inputs=["estimation_method", "identification_strategy"],
                optional_inputs=["gmm_options", "robustness_checks"]
            ),
            WorkflowStep(
                step_id="parameter_estimation",
                title="Parameter Estimation",
                description="Run three-stage estimation procedure",
                required_inputs=[],
                optional_inputs=[]
            ),
            WorkflowStep(
                step_id="results_review",
                title="Results Review",
                description="Review estimation results and diagnostics",
                required_inputs=[],
                optional_inputs=["export_format"]
            )
        ]
        
        workflows["parameter_estimation"] = AnalysisWorkflow(
            workflow_id="parameter_estimation",
            name="Parameter Estimation",
            description="Estimate regional structural parameters using three-stage procedure",
            workflow_type=WorkflowType.PARAMETER_ESTIMATION,
            steps=param_steps
        )
        
        # Policy Analysis Workflow
        policy_steps = [
            WorkflowStep(
                step_id="parameter_input",
                title="Parameter Input",
                description="Load or input estimated regional parameters",
                required_inputs=["parameter_source"],
                optional_inputs=["parameter_file", "manual_parameters"]
            ),
            WorkflowStep(
                step_id="policy_period",
                title="Policy Period Selection",
                description="Select time period for policy analysis",
                required_inputs=["analysis_start", "analysis_end"],
                optional_inputs=["specific_episodes"]
            ),
            WorkflowStep(
                step_id="fed_weights",
                title="Fed Weight Estimation",
                description="Estimate Fed's implicit regional weights",
                required_inputs=["reaction_function_spec"],
                optional_inputs=["instrument_variables"]
            ),
            WorkflowStep(
                step_id="mistake_decomposition",
                title="Policy Mistake Decomposition",
                description="Decompose policy mistakes into components",
                required_inputs=[],
                optional_inputs=["decomposition_method"]
            ),
            WorkflowStep(
                step_id="policy_results",
                title="Policy Analysis Results",
                description="Review policy analysis results and visualizations",
                required_inputs=[],
                optional_inputs=["visualization_options"]
            )
        ]
        
        workflows["policy_analysis"] = AnalysisWorkflow(
            workflow_id="policy_analysis",
            name="Policy Analysis",
            description="Analyze monetary policy effectiveness and decompose policy mistakes",
            workflow_type=WorkflowType.POLICY_ANALYSIS,
            steps=policy_steps
        )
        
        # Counterfactual Analysis Workflow
        counterfactual_steps = [
            WorkflowStep(
                step_id="baseline_setup",
                title="Baseline Scenario Setup",
                description="Configure baseline policy scenario",
                required_inputs=["baseline_period", "baseline_policy"],
                optional_inputs=["baseline_assumptions"]
            ),
            WorkflowStep(
                step_id="alternative_scenarios",
                title="Alternative Scenarios",
                description="Define alternative policy scenarios",
                required_inputs=["scenario_types"],
                optional_inputs=["custom_scenarios"]
            ),
            WorkflowStep(
                step_id="welfare_function",
                title="Welfare Function Setup",
                description="Configure social welfare function parameters",
                required_inputs=["welfare_weights", "preference_parameters"],
                optional_inputs=["welfare_function_type"]
            ),
            WorkflowStep(
                step_id="counterfactual_computation",
                title="Counterfactual Computation",
                description="Compute counterfactual scenarios and welfare outcomes",
                required_inputs=[],
                optional_inputs=["parallel_processing"]
            ),
            WorkflowStep(
                step_id="welfare_comparison",
                title="Welfare Comparison",
                description="Compare welfare outcomes across scenarios",
                required_inputs=[],
                optional_inputs=["comparison_metrics"]
            )
        ]
        
        workflows["counterfactual_analysis"] = AnalysisWorkflow(
            workflow_id="counterfactual_analysis",
            name="Counterfactual Analysis",
            description="Evaluate welfare implications of alternative monetary policies",
            workflow_type=WorkflowType.COUNTERFACTUAL_ANALYSIS,
            steps=counterfactual_steps
        )
        
        return workflows
    
    def get_workflow_by_type(self, workflow_type: WorkflowType) -> List[AnalysisWorkflow]:
        """Get workflows by type.
        
        Args:
            workflow_type: Type of workflow
            
        Returns:
            List of workflows of specified type
        """
        return [w for w in self.workflows.values() if w.workflow_type == workflow_type]