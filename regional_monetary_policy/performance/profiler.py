"""
Performance profiler for identifying computational bottlenecks.
"""

import time
import cProfile
import pstats
import io
import functools
import psutil
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    """Container for profiling results."""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    call_count: int
    bottlenecks: List[str]

class PerformanceProfiler:
    """Profiles performance of estimation procedures and identifies bottlenecks."""
    
    def __init__(self):
        self.profile_results: Dict[str, ProfileResult] = {}
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function performance."""
        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_execution(name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _profile_execution(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with profiling."""
        # Start profiling
        profiler = cProfile.Profile()
        process = psutil.Process()
        
        # Baseline measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        # Execute with profiling
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # End measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        
        # Analyze profiling results
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = max(end_cpu - start_cpu, 0)
        
        # Extract bottlenecks from profiler
        bottlenecks = self._extract_bottlenecks(profiler)
        
        # Store results
        self.profile_results[name] = ProfileResult(
            function_name=name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            call_count=self.profile_results.get(name, ProfileResult("", 0, 0, 0, 0, [])).call_count + 1,
            bottlenecks=bottlenecks
        )
        
        logger.info(f"Profiled {name}: {execution_time:.3f}s, {memory_usage:.1f}MB, {cpu_usage:.1f}% CPU")
        
        return result
    
    def _extract_bottlenecks(self, profiler: cProfile.Profile) -> List[str]:
        """Extract performance bottlenecks from profiler."""
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        # Parse output to identify bottlenecks
        output = s.getvalue()
        lines = output.split('\n')
        bottlenecks = []
        
        for line in lines:
            if 'function calls' in line or 'seconds' in line:
                continue
            if line.strip() and not line.startswith('ncalls'):
                parts = line.split()
                if len(parts) >= 6:
                    # Extract function name and cumulative time
                    cumtime = float(parts[3]) if parts[3].replace('.', '').isdigit() else 0
                    if cumtime > 0.01:  # Functions taking more than 10ms
                        func_name = parts[-1] if parts else "unknown"
                        bottlenecks.append(f"{func_name}: {cumtime:.3f}s")
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager for profiling code blocks."""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            logger.info(f"Block {block_name}: {execution_time:.3f}s, {memory_usage:.1f}MB")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling results."""
        if not self.profile_results:
            return {"message": "No profiling data available"}
        
        total_time = sum(r.execution_time for r in self.profile_results.values())
        total_memory = sum(r.memory_usage for r in self.profile_results.values())
        
        # Find slowest functions
        slowest = sorted(self.profile_results.items(), 
                        key=lambda x: x[1].execution_time, reverse=True)[:5]
        
        # Find memory-intensive functions
        memory_intensive = sorted(self.profile_results.items(),
                                key=lambda x: x[1].memory_usage, reverse=True)[:5]
        
        return {
            "total_execution_time": total_time,
            "total_memory_usage": total_memory,
            "function_count": len(self.profile_results),
            "slowest_functions": [(name, result.execution_time) for name, result in slowest],
            "memory_intensive_functions": [(name, result.memory_usage) for name, result in memory_intensive],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        for name, result in self.profile_results.items():
            if result.execution_time > 10.0:  # Functions taking more than 10 seconds
                recommendations.append(f"Consider optimizing {name} - takes {result.execution_time:.1f}s")
            
            if result.memory_usage > 1000:  # Functions using more than 1GB
                recommendations.append(f"Consider memory optimization for {name} - uses {result.memory_usage:.1f}MB")
            
            if len(result.bottlenecks) > 3:
                recommendations.append(f"Multiple bottlenecks detected in {name} - review implementation")
        
        if not recommendations:
            recommendations.append("Performance looks good - no major bottlenecks detected")
        
        return recommendations
    
    def export_profile_report(self, filepath: str):
        """Export detailed profiling report."""
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            f.write("# Performance Profile Report\n\n")
            f.write(f"**Total Execution Time:** {summary['total_execution_time']:.3f} seconds\n")
            f.write(f"**Total Memory Usage:** {summary['total_memory_usage']:.1f} MB\n")
            f.write(f"**Functions Profiled:** {summary['function_count']}\n\n")
            
            f.write("## Slowest Functions\n")
            for name, time_taken in summary['slowest_functions']:
                f.write(f"- {name}: {time_taken:.3f}s\n")
            
            f.write("\n## Memory Intensive Functions\n")
            for name, memory in summary['memory_intensive_functions']:
                f.write(f"- {name}: {memory:.1f}MB\n")
            
            f.write("\n## Recommendations\n")
            for rec in summary['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write("\n## Detailed Results\n")
            for name, result in self.profile_results.items():
                f.write(f"\n### {name}\n")
                f.write(f"- Execution Time: {result.execution_time:.3f}s\n")
                f.write(f"- Memory Usage: {result.memory_usage:.1f}MB\n")
                f.write(f"- CPU Usage: {result.cpu_usage:.1f}%\n")
                f.write(f"- Call Count: {result.call_count}\n")
                if result.bottlenecks:
                    f.write("- Bottlenecks:\n")
                    for bottleneck in result.bottlenecks:
                        f.write(f"  - {bottleneck}\n")