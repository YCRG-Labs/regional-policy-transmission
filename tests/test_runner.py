"""
Comprehensive test runner for the regional monetary policy analysis system.

This module provides utilities for running different test suites and
generating comprehensive test reports.
"""

import pytest
import sys
import time
from pathlib import Path
import json
import subprocess
from typing import Dict, List, Optional
import warnings


class TestSuiteRunner_:
    """Comprehensive test suite runner with reporting capabilities."""
    
    def __init__(self, test_dir: Path = None):
        """Initialize test runner."""
        self.test_dir = test_dir or Path("tests")
        self.results = {}
        
    def run_unit_tests(self, verbose: bool = True) -> Dict:
        """Run unit tests for core mathematical operations and algorithms."""
        print("Running Unit Tests...")
        
        unit_test_files = [
            "test_mathematical_operations.py",
            "test_parameter_estimator.py",
            "test_spatial_handler.py",
            "test_policy_analysis.py",
            "test_counterfactual_engine.py",
            "test_data_manager.py",
            "test_fred_client.py"
        ]
        
        return self._run_test_subset(unit_test_files, "unit", verbose)
    
    def run_integration_tests(self, verbose: bool = True) -> Dict:
        """Run integration tests for complete analysis workflows."""
        print("Running Integration Tests...")
        
        integration_test_files = [
            "test_integration_workflows.py"
        ]
        
        return self._run_test_subset(integration_test_files, "integration", verbose)
    
    def run_validation_tests(self, verbose: bool = True) -> Dict:
        """Run validation tests using synthetic data and known solutions."""
        print("Running Validation Tests...")
        
        validation_test_files = [
            "test_synthetic_validation.py"
        ]
        
        return self._run_test_subset(validation_test_files, "validation", verbose)
    
    def run_performance_tests(self, verbose: bool = True) -> Dict:
        """Run performance benchmarks and regression tests."""
        print("Running Performance Tests...")
        
        performance_test_files = [
            "test_performance_benchmarks.py"
        ]
        
        return self._run_test_subset(performance_test_files, "performance", verbose)
    
    def run_replication_tests(self, verbose: bool = True) -> Dict:
        """Run replication tests using published research results."""
        print("Running Replication Tests...")
        
        replication_test_files = [
            "test_replication_studies.py"
        ]
        
        return self._run_test_subset(replication_test_files, "replication", verbose)
    
    def run_all_tests(self, verbose: bool = True, include_slow: bool = False) -> Dict:
        """Run all test suites."""
        print("Running Complete Test Suite...")
        
        start_time = time.time()
        
        # Configure pytest arguments
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "--color=yes"
        ]
        
        if not include_slow:
            pytest_args.extend(["-m", "not slow"])
        
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        end_time = time.time()
        
        result = {
            "suite": "all",
            "exit_code": exit_code,
            "duration": end_time - start_time,
            "success": exit_code == 0
        }
        
        self.results["all"] = result
        return result
    
    def run_quick_tests(self, verbose: bool = True) -> Dict:
        """Run quick test suite (excluding slow tests)."""
        print("Running Quick Test Suite...")
        
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "not slow",
            "--tb=short",
            "--color=yes"
        ]
        
        start_time = time.time()
        exit_code = pytest.main(pytest_args)
        end_time = time.time()
        
        result = {
            "suite": "quick",
            "exit_code": exit_code,
            "duration": end_time - start_time,
            "success": exit_code == 0
        }
        
        self.results["quick"] = result
        return result
    
    def _run_test_subset(self, test_files: List[str], suite_name: str, verbose: bool) -> Dict:
        """Run a subset of test files."""
        start_time = time.time()
        
        # Build pytest arguments
        pytest_args = []
        for test_file in test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                pytest_args.append(str(test_path))
        
        if not pytest_args:
            print(f"No test files found for {suite_name} suite")
            return {"suite": suite_name, "exit_code": 1, "duration": 0, "success": False}
        
        pytest_args.extend([
            "-v" if verbose else "-q",
            "--tb=short",
            "--color=yes"
        ])
        
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        end_time = time.time()
        
        result = {
            "suite": suite_name,
            "exit_code": exit_code,
            "duration": end_time - start_time,
            "success": exit_code == 0,
            "test_files": test_files
        }
        
        self.results[suite_name] = result
        return result
    
    def generate_test_report(self, output_file: Optional[Path] = None) -> Dict:
        """Generate comprehensive test report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_suites": len(self.results),
            "successful_suites": sum(1 for r in self.results.values() if r["success"]),
            "total_duration": sum(r["duration"] for r in self.results.values()),
            "suite_results": self.results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Test report saved to {output_file}")
        
        return report
    
    def print_summary(self):
        """Print test results summary."""
        if not self.results:
            print("No test results available")
            return
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        total_duration = 0
        successful_suites = 0
        
        for suite_name, result in self.results.items():
            status = "PASSED" if result["success"] else "FAILED"
            duration = result["duration"]
            total_duration += duration
            
            if result["success"]:
                successful_suites += 1
            
            print(f"{suite_name.upper():20} {status:8} ({duration:.2f}s)")
        
        print("-" * 60)
        print(f"{'TOTAL':20} {successful_suites}/{len(self.results)} suites passed ({total_duration:.2f}s)")
        
        if successful_suites == len(self.results):
            print("\n✅ All test suites passed!")
        else:
            print(f"\n❌ {len(self.results) - successful_suites} test suite(s) failed")


def run_test_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        runner = TestSuiteRunner()
        result = runner.run_all_tests(verbose=False)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print("\nGenerating coverage report...")
        cov.report()
        
        # Generate HTML coverage report
        cov.html_report(directory="htmlcov")
        print("HTML coverage report generated in 'htmlcov' directory")
        
        return result
        
    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")
        return {"success": False, "error": "Coverage package not available"}


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regional Monetary Policy Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "validation", "performance", "replication", "all", "quick"],
                       default="quick", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--report", type=str, help="Save test report to file")
    
    args = parser.parse_args()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    runner = TestSuiteRunner()
    
    if args.coverage:
        result = run_test_coverage()
    else:
        # Run specified test suite
        if args.suite == "unit":
            result = runner.run_unit_tests(args.verbose)
        elif args.suite == "integration":
            result = runner.run_integration_tests(args.verbose)
        elif args.suite == "validation":
            result = runner.run_validation_tests(args.verbose)
        elif args.suite == "performance":
            result = runner.run_performance_tests(args.verbose)
        elif args.suite == "replication":
            result = runner.run_replication_tests(args.verbose)
        elif args.suite == "all":
            result = runner.run_all_tests(args.verbose, args.include_slow)
        elif args.suite == "quick":
            result = runner.run_quick_tests(args.verbose)
    
    # Print summary
    runner.print_summary()
    
    # Generate report if requested
    if args.report:
        runner.generate_test_report(Path(args.report))
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()