#!/usr/bin/env python3
"""
Full Pipeline Runner for Regional Monetary Policy Analysis System

This script runs the complete analysis pipeline including:
1. System validation and setup
2. Data retrieval and validation
3. Spatial modeling and parameter estimation
4. Policy analysis and mistake decomposition
5. Counterfactual analysis
6. Visualization and reporting

Usage:
    python scripts/run_full_pipeline.py [options]

Options:
    --start-date YYYY-MM-DD    Start date for analysis (default: 2000-01-01)
    --end-date YYYY-MM-DD      End date for analysis (default: 2023-12-31)
    --regions REGION1,REGION2  Comma-separated list of regions (default: from config)
    --config CONFIG_PATH       Path to configuration file (default: None)
    --analysis-name NAME       Name for this analysis run (default: auto-generated)
    --quick                    Run quick analysis for testing (default: False)
    --validate-only            Only run system validation (default: False)
    --output-dir DIR           Output directory for results (default: output/)
    --verbose                  Enable verbose logging (default: False)
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
from pathlib import Path
import traceback
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from regional_monetary_policy.integration.workflow_engine import WorkflowEngine
from regional_monetary_policy.integration.system_validator import SystemValidator
from regional_monetary_policy.integration.pipeline_manager import PipelineManager, PipelineStep
from regional_monetary_policy.exceptions import RegionalMonetaryPolicyError
from regional_monetary_policy.logging_config import setup_logging


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run complete Regional Monetary Policy Analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2000-01-01',
        help='Start date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str,
        default='2023-12-31',
        help='End date for analysis (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--regions',
        type=str,
        help='Comma-separated list of regions (e.g., CA,TX,NY)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--analysis-name',
        type=str,
        help='Name for this analysis run'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick analysis for testing'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run system validation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def validate_date_format(date_string):
    """Validate date format and return datetime object."""
    try:
        return datetime.strptime(date_string, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD format.")


def create_output_directory(output_dir):
    """Create output directory structure."""
    output_path = Path(output_dir)
    
    # Create main output directory
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ['data', 'figures', 'reports', 'tables', 'logs']
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)
    
    return output_path


def run_system_validation(config_path=None, verbose=False):
    """Run comprehensive system validation."""
    print("🔍 SYSTEM VALIDATION")
    print("=" * 50)
    
    try:
        validator = SystemValidator(config_path)
        
        print("Running comprehensive system validation...")
        validation_results = validator.run_comprehensive_validation()
        
        # Print validation summary
        overall_status = validation_results.get('overall_status', 'unknown')
        print(f"\nOverall Status: {'✅ PASSED' if overall_status == 'passed' else '❌ FAILED'}")
        
        # Print component results
        component_tests = validation_results.get('component_tests', {})
        if component_tests:
            print(f"\nComponent Tests: {component_tests.get('passed', 0)}/{component_tests.get('total', 0)} passed")
        
        # Print integration results
        integration_tests = validation_results.get('integration_tests', {})
        if integration_tests:
            print(f"Integration Tests: {integration_tests.get('passed', 0)}/{integration_tests.get('total', 0)} passed")
        
        # Print performance results
        performance_tests = validation_results.get('performance_tests', {})
        if performance_tests:
            print(f"Performance Tests: {performance_tests.get('passed', 0)}/{performance_tests.get('total', 0)} passed")
        
        if verbose:
            print("\nDetailed Results:")
            print(json.dumps(validation_results, indent=2, default=str))
        
        return overall_status == 'passed'
        
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_quick_analysis(workflow_engine, start_date, end_date, verbose=False):
    """Run quick analysis for testing."""
    print("\n🚀 QUICK ANALYSIS")
    print("=" * 50)
    
    try:
        print(f"Running quick analysis from {start_date} to {end_date}...")
        
        results = workflow_engine.run_quick_analysis(start_date, end_date)
        
        print("✅ Quick analysis completed successfully!")
        print(f"Data shape: {results.get('data_shape', 'N/A')}")
        print(f"Regions analyzed: {results.get('regions_analyzed', 'N/A')}")
        print(f"Parameters estimated: {'✅' if results.get('parameters_estimated') else '❌'}")
        print(f"Optimal policy computed: {'✅' if results.get('optimal_policy_computed') else '❌'}")
        print(f"Spatial weights shape: {results.get('spatial_weights_shape', 'N/A')}")
        
        if verbose:
            print("\nDetailed Results:")
            print(json.dumps(results, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_complete_analysis(workflow_engine, start_date, end_date, regions=None, 
                         analysis_name=None, output_dir=None, verbose=False):
    """Run complete end-to-end analysis."""
    print("\n🎯 COMPLETE ANALYSIS")
    print("=" * 50)
    
    try:
        if analysis_name is None:
            analysis_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Analysis Name: {analysis_name}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Regions: {regions if regions else 'Default from config'}")
        print(f"Output Directory: {output_dir}")
        
        print("\nStarting complete analysis workflow...")
        
        results = workflow_engine.run_complete_analysis(
            start_date=start_date,
            end_date=end_date,
            regions=regions,
            analysis_name=analysis_name
        )
        
        print("✅ Complete analysis finished successfully!")
        
        # Print summary results
        print(f"\nAnalysis Summary:")
        print(f"  - Analysis Name: {results.get('analysis_name')}")
        print(f"  - Timestamp: {results.get('timestamp')}")
        print(f"  - Report Path: {results.get('report_path', 'N/A')}")
        
        # Save results to output directory if specified
        if output_dir:
            results_file = Path(output_dir) / 'reports' / f"{analysis_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  - Results saved to: {results_file}")
        
        if verbose:
            print("\nDetailed Results:")
            print(json.dumps(results, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Complete analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def run_pipeline_analysis(start_date, end_date, regions=None, config_path=None,
                         analysis_name=None, output_dir=None, verbose=False):
    """Run analysis using pipeline manager for more control."""
    print("\n⚙️ PIPELINE ANALYSIS")
    print("=" * 50)
    
    try:
        # Initialize pipeline manager
        pipeline_manager = PipelineManager(max_workers=4)
        
        # Create analysis pipeline steps
        def data_retrieval_step(**kwargs):
            print("  📊 Retrieving and preparing data...")
            workflow_engine = kwargs.get('workflow_engine')
            return workflow_engine._retrieve_and_prepare_data(start_date, end_date, regions)
        
        def parameter_estimation_step(**kwargs):
            print("  📈 Estimating parameters...")
            workflow_engine = kwargs.get('workflow_engine')
            regional_data = kwargs.get('data_retrieval_result')
            return workflow_engine._estimate_parameters(regional_data)
        
        def policy_analysis_step(**kwargs):
            print("  🏛️ Analyzing policy mistakes...")
            workflow_engine = kwargs.get('workflow_engine')
            regional_data = kwargs.get('data_retrieval_result')
            spatial_weights, regional_params = kwargs.get('parameter_estimation_result')
            return workflow_engine._analyze_policy_mistakes(regional_data, regional_params)
        
        def counterfactual_step(**kwargs):
            print("  🔄 Running counterfactual analysis...")
            workflow_engine = kwargs.get('workflow_engine')
            regional_data = kwargs.get('data_retrieval_result')
            spatial_weights, regional_params = kwargs.get('parameter_estimation_result')
            return workflow_engine._run_counterfactual_analysis(regional_data, regional_params)
        
        def visualization_step(**kwargs):
            print("  📊 Generating visualizations...")
            workflow_engine = kwargs.get('workflow_engine')
            regional_data = kwargs.get('data_retrieval_result')
            spatial_weights, regional_params = kwargs.get('parameter_estimation_result')
            policy_results = kwargs.get('policy_analysis_result')
            counterfactual_results = kwargs.get('counterfactual_step_result')
            return workflow_engine._generate_visualizations(
                regional_data, regional_params, policy_results, counterfactual_results
            )
        
        def report_generation_step(**kwargs):
            print("  📄 Generating comprehensive report...")
            workflow_engine = kwargs.get('workflow_engine')
            regional_data = kwargs.get('data_retrieval_result')
            spatial_weights, regional_params = kwargs.get('parameter_estimation_result')
            policy_results = kwargs.get('policy_analysis_result')
            counterfactual_results = kwargs.get('counterfactual_step_result')
            visualizations = kwargs.get('visualization_step_result')
            
            if analysis_name is None:
                name = f"pipeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                name = analysis_name
                
            return workflow_engine._generate_comprehensive_report(
                name, regional_data, regional_params, policy_results,
                counterfactual_results, visualizations
            )
        
        # Define pipeline steps
        steps = [
            PipelineStep(
                name="data_retrieval",
                function=data_retrieval_step,
                dependencies=[]
            ),
            PipelineStep(
                name="parameter_estimation",
                function=parameter_estimation_step,
                dependencies=["data_retrieval"]
            ),
            PipelineStep(
                name="policy_analysis",
                function=policy_analysis_step,
                dependencies=["data_retrieval", "parameter_estimation"],
                parallel=True
            ),
            PipelineStep(
                name="counterfactual_step",
                function=counterfactual_step,
                dependencies=["data_retrieval", "parameter_estimation"],
                parallel=True
            ),
            PipelineStep(
                name="visualization_step",
                function=visualization_step,
                dependencies=["data_retrieval", "parameter_estimation", 
                            "policy_analysis", "counterfactual_step"]
            ),
            PipelineStep(
                name="report_generation",
                function=report_generation_step,
                dependencies=["data_retrieval", "parameter_estimation", 
                            "policy_analysis", "counterfactual_step", "visualization_step"]
            )
        ]
        
        # Register and execute pipeline
        pipeline_name = "monetary_policy_analysis"
        pipeline_manager.register_pipeline(pipeline_name, steps)
        
        # Initialize workflow engine for pipeline steps
        workflow_engine = WorkflowEngine(config_path)
        
        print("Executing analysis pipeline...")
        results = pipeline_manager.execute_pipeline(
            pipeline_name, 
            pipeline_args={'workflow_engine': workflow_engine}
        )
        
        # Check results
        successful_steps = sum(1 for result in results.values() 
                             if result.status.value == 'completed')
        total_steps = len(results)
        
        print(f"✅ Pipeline completed: {successful_steps}/{total_steps} steps successful")
        
        # Print step results
        for step_name, result in results.items():
            status_icon = "✅" if result.status.value == "completed" else "❌"
            duration = f"{result.duration:.2f}s" if result.duration else "N/A"
            print(f"  {status_icon} {step_name}: {result.status.value} ({duration})")
        
        return successful_steps == total_steps
        
    except Exception as e:
        print(f"❌ Pipeline analysis failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    """Main function to run the full pipeline."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    print("🏛️ REGIONAL MONETARY POLICY ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Validate dates
        start_date = validate_date_format(args.start_date)
        end_date = validate_date_format(args.end_date)
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Parse regions
        regions = None
        if args.regions:
            regions = [r.strip() for r in args.regions.split(',')]
        
        # Create output directory
        output_dir = create_output_directory(args.output_dir)
        print(f"Output directory: {output_dir.absolute()}")
        
        # Step 1: System Validation
        if not run_system_validation(args.config, args.verbose):
            print("\n❌ System validation failed. Please fix issues before proceeding.")
            return 1
        
        if args.validate_only:
            print("\n✅ Validation complete. Exiting as requested.")
            return 0
        
        # Initialize workflow engine
        print("\n🔧 SYSTEM INITIALIZATION")
        print("=" * 50)
        print("Initializing workflow engine...")
        
        workflow_engine = WorkflowEngine(args.config)
        print("✅ Workflow engine initialized successfully")
        
        # Step 2: Run Analysis
        success = False
        
        if args.quick:
            success = run_quick_analysis(
                workflow_engine, args.start_date, args.end_date, args.verbose
            )
        else:
            # Try complete analysis first
            try:
                success = run_complete_analysis(
                    workflow_engine, args.start_date, args.end_date, regions,
                    args.analysis_name, output_dir, args.verbose
                )
            except Exception as e:
                print(f"Complete analysis failed: {e}")
                print("Falling back to pipeline analysis...")
                
                success = run_pipeline_analysis(
                    args.start_date, args.end_date, regions, args.config,
                    args.analysis_name, output_dir, args.verbose
                )
        
        # Final summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        if success:
            print("🎉 Pipeline completed successfully!")
            print(f"Results available in: {output_dir.absolute()}")
        else:
            print("❌ Pipeline execution failed")
            print("Check the logs above for detailed error information")
        
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())