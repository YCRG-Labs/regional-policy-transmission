"""
Pipeline Manager for coordinating analysis pipelines and managing execution flow.

This module provides utilities for managing complex analysis pipelines,
handling dependencies, and coordinating parallel execution.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

from ..exceptions import RegionalMonetaryPolicyError
from ..progress_monitor import ProgressMonitor

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    """Represents a single step in an analysis pipeline."""
    name: str
    function: Callable
    dependencies: List[str]
    args: Tuple = ()
    kwargs: Dict[str, Any] = None
    parallel: bool = False
    timeout: Optional[int] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class PipelineResult:
    """Result of pipeline step execution."""
    step_name: str
    status: PipelineStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None


class PipelineManager:
    """
    Manages execution of complex analysis pipelines with dependency resolution.
    
    This class provides functionality for:
    - Defining multi-step analysis pipelines
    - Managing step dependencies
    - Parallel execution where possible
    - Progress monitoring and error handling
    - Result caching and retrieval
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize pipeline manager.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.pipelines: Dict[str, List[PipelineStep]] = {}
        self.results: Dict[str, Dict[str, PipelineResult]] = {}
        self.progress_monitor = ProgressMonitor()
        self._lock = threading.Lock()
        
    def register_pipeline(self, pipeline_name: str, steps: List[PipelineStep]):
        """
        Register a new analysis pipeline.
        
        Args:
            pipeline_name: Unique name for the pipeline
            steps: List of pipeline steps
        """
        # Validate pipeline structure
        self._validate_pipeline(steps)
        
        with self._lock:
            self.pipelines[pipeline_name] = steps
            
        logger.info(f"Registered pipeline '{pipeline_name}' with {len(steps)} steps")
    
    def _validate_pipeline(self, steps: List[PipelineStep]):
        """Validate pipeline structure and dependencies."""
        step_names = {step.name for step in steps}
        
        for step in steps:
            # Check for circular dependencies
            if step.name in step.dependencies:
                raise RegionalMonetaryPolicyError(
                    f"Step '{step.name}' cannot depend on itself"
                )
            
            # Check that all dependencies exist
            for dep in step.dependencies:
                if dep not in step_names:
                    raise RegionalMonetaryPolicyError(
                        f"Step '{step.name}' depends on non-existent step '{dep}'"
                    )
        
        # Check for circular dependencies using topological sort
        self._check_circular_dependencies(steps)
    
    def _check_circular_dependencies(self, steps: List[PipelineStep]):
        """Check for circular dependencies using topological sort."""
        # Build dependency graph
        graph = {step.name: step.dependencies for step in steps}
        
        # Kahn's algorithm for topological sort
        in_degree = {name: len(deps) for name, deps in graph.items()}
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        processed = []
        
        while queue:
            current = queue.pop(0)
            processed.append(current)
            
            # For each step that depends on current, reduce its in-degree
            for name, deps in graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        if len(processed) != len(steps):
            raise RegionalMonetaryPolicyError("Circular dependency detected in pipeline")
    
    def execute_pipeline(self, pipeline_name: str, 
                        pipeline_args: Optional[Dict[str, Any]] = None) -> Dict[str, PipelineResult]:
        """
        Execute a registered pipeline.
        
        Args:
            pipeline_name: Name of pipeline to execute
            pipeline_args: Global arguments to pass to all steps
            
        Returns:
            Dictionary mapping step names to their results
        """
        if pipeline_name not in self.pipelines:
            raise RegionalMonetaryPolicyError(f"Pipeline '{pipeline_name}' not registered")
        
        if pipeline_args is None:
            pipeline_args = {}
        
        logger.info(f"Executing pipeline '{pipeline_name}'")
        
        steps = self.pipelines[pipeline_name]
        results = {}
        
        # Initialize progress monitoring
        tracker = self.progress_monitor.create_tracker(
            f"pipeline_{pipeline_name}", 
            total_steps=len(steps)
        )
        
        try:
            # Execute steps in dependency order
            execution_order = self._get_execution_order(steps)
            
            for step_name in execution_order:
                step = next(s for s in steps if s.name == step_name)
                
                # Check if dependencies are satisfied
                if not self._dependencies_satisfied(step, results):
                    raise RegionalMonetaryPolicyError(
                        f"Dependencies not satisfied for step '{step_name}'"
                    )
                
                # Execute step
                result = self._execute_step(step, results, pipeline_args)
                results[step_name] = result
                
                # Update progress
                tracker.update(increment=True)
                
                # Stop if step failed and not configured to continue
                if result.status == PipelineStatus.FAILED:
                    logger.error(f"Pipeline step '{step_name}' failed: {result.error}")
                    break
            
            # Store results
            with self._lock:
                self.results[pipeline_name] = results
            
            logger.info(f"Pipeline '{pipeline_name}' execution completed")
            
        except Exception as e:
            logger.error(f"Pipeline '{pipeline_name}' execution failed: {e}")
            raise RegionalMonetaryPolicyError(f"Pipeline execution failed: {e}")
        
        finally:
            tracker.complete()
        
        return results
    
    def _get_execution_order(self, steps: List[PipelineStep]) -> List[str]:
        """Get execution order based on dependencies."""
        # Topological sort to determine execution order
        graph = {step.name: step.dependencies for step in steps}
        in_degree = {name: len(deps) for name, deps in graph.items()}
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # For each step that depends on current, reduce its in-degree
            for name, deps in graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        return order
    
    def _dependencies_satisfied(self, step: PipelineStep, 
                              results: Dict[str, PipelineResult]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        for dep in step.dependencies:
            if dep not in results or results[dep].status != PipelineStatus.COMPLETED:
                return False
        return True
    
    def _execute_step(self, step: PipelineStep, 
                     previous_results: Dict[str, PipelineResult],
                     pipeline_args: Dict[str, Any]) -> PipelineResult:
        """Execute a single pipeline step."""
        logger.info(f"Executing step '{step.name}'")
        
        result = PipelineResult(
            step_name=step.name,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Prepare arguments
            args = step.args
            kwargs = {**step.kwargs, **pipeline_args}
            
            # Add dependency results to kwargs
            for dep in step.dependencies:
                if dep in previous_results:
                    kwargs[f"{dep}_result"] = previous_results[dep].result
            
            # Execute step function
            if step.parallel and self.max_workers > 1:
                # Execute in parallel if configured
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(step.function, *args, **kwargs)
                    if step.timeout:
                        step_result = future.result(timeout=step.timeout)
                    else:
                        step_result = future.result()
            else:
                # Execute synchronously
                step_result = step.function(*args, **kwargs)
            
            result.result = step_result
            result.status = PipelineStatus.COMPLETED
            
            logger.info(f"Step '{step.name}' completed successfully")
            
        except Exception as e:
            result.error = e
            result.status = PipelineStatus.FAILED
            logger.error(f"Step '{step.name}' failed: {e}")
        
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def execute_parallel_pipeline(self, pipeline_name: str,
                                pipeline_args: Optional[Dict[str, Any]] = None) -> Dict[str, PipelineResult]:
        """
        Execute pipeline with maximum parallelization where possible.
        
        Args:
            pipeline_name: Name of pipeline to execute
            pipeline_args: Global arguments to pass to all steps
            
        Returns:
            Dictionary mapping step names to their results
        """
        if pipeline_name not in self.pipelines:
            raise RegionalMonetaryPolicyError(f"Pipeline '{pipeline_name}' not registered")
        
        if pipeline_args is None:
            pipeline_args = {}
        
        logger.info(f"Executing parallel pipeline '{pipeline_name}'")
        
        steps = self.pipelines[pipeline_name]
        results = {}
        completed_steps = set()
        
        # Initialize progress monitoring
        tracker = self.progress_monitor.create_tracker(
            f"parallel_pipeline_{pipeline_name}", 
            total_steps=len(steps)
        )
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                while len(completed_steps) < len(steps):
                    # Find steps ready to execute
                    ready_steps = []
                    for step in steps:
                        if (step.name not in completed_steps and 
                            step.name not in futures and
                            all(dep in completed_steps for dep in step.dependencies)):
                            ready_steps.append(step)
                    
                    # Submit ready steps for execution
                    for step in ready_steps:
                        future = executor.submit(
                            self._execute_step, step, results, pipeline_args
                        )
                        futures[step.name] = future
                    
                    # Wait for at least one step to complete
                    if futures:
                        # Check for completed futures
                        completed_futures = []
                        for step_name, future in futures.items():
                            if future.done():
                                result = future.result()
                                results[step_name] = result
                                completed_steps.add(step_name)
                                completed_futures.append(step_name)
                                
                                # Update progress
                                tracker.update(increment=True)
                        
                        # Remove completed futures
                        for step_name in completed_futures:
                            del futures[step_name]
                    
                    # Small delay to prevent busy waiting
                    if not completed_futures and futures:
                        import time
                        time.sleep(0.1)
            
            # Store results
            with self._lock:
                self.results[pipeline_name] = results
            
            logger.info(f"Parallel pipeline '{pipeline_name}' execution completed")
            
        except Exception as e:
            logger.error(f"Parallel pipeline '{pipeline_name}' execution failed: {e}")
            raise RegionalMonetaryPolicyError(f"Parallel pipeline execution failed: {e}")
        
        finally:
            tracker.complete()
        
        return results
    
    def get_pipeline_results(self, pipeline_name: str) -> Optional[Dict[str, PipelineResult]]:
        """Get results from a previously executed pipeline."""
        with self._lock:
            return self.results.get(pipeline_name)
    
    def get_step_result(self, pipeline_name: str, step_name: str) -> Optional[PipelineResult]:
        """Get result from a specific pipeline step."""
        pipeline_results = self.get_pipeline_results(pipeline_name)
        if pipeline_results:
            return pipeline_results.get(step_name)
        return None
    
    def clear_results(self, pipeline_name: Optional[str] = None):
        """Clear stored results for a pipeline or all pipelines."""
        with self._lock:
            if pipeline_name:
                self.results.pop(pipeline_name, None)
            else:
                self.results.clear()
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get current status of a pipeline."""
        if pipeline_name not in self.pipelines:
            return {'status': 'not_registered'}
        
        results = self.get_pipeline_results(pipeline_name)
        if not results:
            return {'status': 'not_executed'}
        
        step_statuses = {name: result.status.value for name, result in results.items()}
        
        # Determine overall status
        if all(status == PipelineStatus.COMPLETED.value for status in step_statuses.values()):
            overall_status = 'completed'
        elif any(status == PipelineStatus.FAILED.value for status in step_statuses.values()):
            overall_status = 'failed'
        elif any(status == PipelineStatus.RUNNING.value for status in step_statuses.values()):
            overall_status = 'running'
        else:
            overall_status = 'unknown'
        
        return {
            'status': overall_status,
            'step_statuses': step_statuses,
            'total_steps': len(self.pipelines[pipeline_name]),
            'completed_steps': sum(1 for s in step_statuses.values() 
                                 if s == PipelineStatus.COMPLETED.value)
        }
    
    def create_standard_analysis_pipeline(self) -> List[PipelineStep]:
        """Create a standard analysis pipeline with common steps."""
        return [
            PipelineStep(
                name="data_retrieval",
                function=self._dummy_data_retrieval,
                dependencies=[],
                parallel=False
            ),
            PipelineStep(
                name="data_validation",
                function=self._dummy_data_validation,
                dependencies=["data_retrieval"],
                parallel=False
            ),
            PipelineStep(
                name="spatial_modeling",
                function=self._dummy_spatial_modeling,
                dependencies=["data_validation"],
                parallel=False
            ),
            PipelineStep(
                name="parameter_estimation",
                function=self._dummy_parameter_estimation,
                dependencies=["spatial_modeling"],
                parallel=True
            ),
            PipelineStep(
                name="policy_analysis",
                function=self._dummy_policy_analysis,
                dependencies=["parameter_estimation"],
                parallel=True
            ),
            PipelineStep(
                name="counterfactual_analysis",
                function=self._dummy_counterfactual_analysis,
                dependencies=["parameter_estimation"],
                parallel=True
            ),
            PipelineStep(
                name="visualization",
                function=self._dummy_visualization,
                dependencies=["policy_analysis", "counterfactual_analysis"],
                parallel=False
            ),
            PipelineStep(
                name="report_generation",
                function=self._dummy_report_generation,
                dependencies=["visualization"],
                parallel=False
            )
        ]
    
    # Dummy functions for pipeline testing
    def _dummy_data_retrieval(self, **kwargs):
        """Dummy data retrieval function."""
        import time
        time.sleep(0.1)  # Simulate work
        return {"data": "retrieved"}
    
    def _dummy_data_validation(self, **kwargs):
        """Dummy data validation function."""
        import time
        time.sleep(0.1)
        return {"validation": "passed"}
    
    def _dummy_spatial_modeling(self, **kwargs):
        """Dummy spatial modeling function."""
        import time
        time.sleep(0.1)
        return {"spatial_weights": "computed"}
    
    def _dummy_parameter_estimation(self, **kwargs):
        """Dummy parameter estimation function."""
        import time
        time.sleep(0.2)
        return {"parameters": "estimated"}
    
    def _dummy_policy_analysis(self, **kwargs):
        """Dummy policy analysis function."""
        import time
        time.sleep(0.1)
        return {"policy_analysis": "completed"}
    
    def _dummy_counterfactual_analysis(self, **kwargs):
        """Dummy counterfactual analysis function."""
        import time
        time.sleep(0.1)
        return {"counterfactual": "completed"}
    
    def _dummy_visualization(self, **kwargs):
        """Dummy visualization function."""
        import time
        time.sleep(0.1)
        return {"visualizations": "generated"}
    
    def _dummy_report_generation(self, **kwargs):
        """Dummy report generation function."""
        import time
        time.sleep(0.1)
        return {"report": "generated"}