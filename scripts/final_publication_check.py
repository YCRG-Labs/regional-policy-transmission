#!/usr/bin/env python3
"""
Final publication readiness check for Regional Monetary Policy Analysis System.
"""

import sys
import os
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_header():
    """Print validation header."""
    print("=" * 80)
    print("FINAL PUBLICATION READINESS CHECK")
    print("Regional Monetary Policy Analysis System")
    print("=" * 80)
    print(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_core_functionality():
    """Check core system functionality."""
    print("🔍 CORE FUNCTIONALITY CHECK")
    print("-" * 50)
    
    issues = []
    
    try:
        # Test imports
        import regional_monetary_policy
        from regional_monetary_policy.data.fred_client import FREDClient
        from regional_monetary_policy.data.sample_data import create_sample_dataset
        from regional_monetary_policy.config.config_manager import ConfigManager
        print("✓ Core imports working")
        
        # Test sample data generation
        sample_data = create_sample_dataset()
        if len(sample_data) > 0:
            print(f"✓ Sample data generation: {len(sample_data)} observations")
        else:
            issues.append("Sample data generation failed")
        
        # Test FRED client
        client = FREDClient()
        if client.validate_connection():
            print("✓ FRED client (using sample data)")
        else:
            issues.append("FRED client validation failed")
        
        # Test configuration
        config = ConfigManager()
        print("✓ Configuration system")
        
    except Exception as e:
        issues.append(f"Core functionality error: {str(e)}")
        print(f"✗ Core functionality failed: {str(e)}")
    
    return len(issues) == 0, issues

def check_documentation():
    """Check documentation completeness."""
    print("\n📚 DOCUMENTATION CHECK")
    print("-" * 50)
    
    required_docs = [
        "README.md",
        "docs/api_documentation.md", 
        "DEPLOYMENT_GUIDE.md",
        "docs/faq.md",
        "docs/troubleshooting_guide.md"
    ]
    
    issues = []
    
    for doc in required_docs:
        if Path(doc).exists():
            size = Path(doc).stat().st_size
            if size > 1000:  # At least 1KB of content
                print(f"✓ {doc} ({size:,} bytes)")
            else:
                issues.append(f"{doc} exists but is too small ({size} bytes)")
                print(f"! {doc} exists but is too small")
        else:
            issues.append(f"Missing documentation: {doc}")
            print(f"✗ Missing: {doc}")
    
    return len(issues) == 0, issues

def check_project_structure():
    """Check project structure completeness."""
    print("\n🏗️ PROJECT STRUCTURE CHECK")
    print("-" * 50)
    
    required_structure = [
        "regional_monetary_policy/__init__.py",
        "regional_monetary_policy/data/",
        "regional_monetary_policy/econometric/",
        "regional_monetary_policy/policy/",
        "regional_monetary_policy/config/",
        "tests/",
        "examples/",
        "docs/",
        "setup.py",
        "requirements.txt"
    ]
    
    issues = []
    
    for item in required_structure:
        path = Path(item)
        if path.exists():
            print(f"✓ {item}")
        else:
            issues.append(f"Missing: {item}")
            print(f"✗ Missing: {item}")
    
    return len(issues) == 0, issues

def check_dependencies():
    """Check dependency installation."""
    print("\n📦 DEPENDENCIES CHECK")
    print("-" * 50)
    
    required_packages = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'plotly',
        'statsmodels', 'sklearn', 'requests', 'pydantic'
    ]
    
    issues = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
            print(f"✗ {package}")
    
    return len(issues) == 0, issues

def check_examples():
    """Check example scripts."""
    print("\n💡 EXAMPLES CHECK")
    print("-" * 50)
    
    example_files = list(Path("examples").glob("*.py")) if Path("examples").exists() else []
    
    if len(example_files) >= 3:
        print(f"✓ Found {len(example_files)} example scripts")
        for example in example_files[:5]:  # Show first 5
            print(f"  - {example.name}")
        return True, []
    else:
        return False, [f"Insufficient examples: {len(example_files)} found, need at least 3"]

def check_tests():
    """Check test coverage."""
    print("\n🧪 TESTS CHECK")
    print("-" * 50)
    
    test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
    
    if len(test_files) >= 5:
        print(f"✓ Found {len(test_files)} test files")
        
        # Try to run a simple test
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '--collect-only', '-q'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✓ Test collection successful")
                return True, []
            else:
                return False, ["Test collection failed"]
        except Exception as e:
            return False, [f"Test execution error: {str(e)}"]
    else:
        return False, [f"Insufficient tests: {len(test_files)} found, need at least 5"]

def assess_publication_readiness():
    """Assess overall publication readiness."""
    print("\n🎯 PUBLICATION READINESS ASSESSMENT")
    print("-" * 50)
    
    # Define criteria and weights
    criteria = [
        ("Core Functionality", check_core_functionality, 0.30),
        ("Documentation", check_documentation, 0.25),
        ("Project Structure", check_project_structure, 0.20),
        ("Dependencies", check_dependencies, 0.10),
        ("Examples", check_examples, 0.10),
        ("Tests", check_tests, 0.05)
    ]
    
    total_score = 0
    max_score = 0
    all_issues = []
    
    for criterion_name, check_func, weight in criteria:
        try:
            passed, issues = check_func()
            score = weight if passed else 0
            total_score += score
            max_score += weight
            
            if issues:
                all_issues.extend([f"{criterion_name}: {issue}" for issue in issues])
                
        except Exception as e:
            print(f"✗ {criterion_name} check failed: {str(e)}")
            all_issues.append(f"{criterion_name}: Check failed - {str(e)}")
    
    # Calculate percentage
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    print(f"\nOverall Score: {percentage:.1f}%")
    
    # Determine readiness level
    if percentage >= 90:
        status = "🟢 PUBLICATION READY"
        recommendation = "System is ready for publication. Minor issues can be addressed post-publication."
    elif percentage >= 75:
        status = "🟡 MOSTLY READY"
        recommendation = "System is mostly ready. Address remaining issues before publication."
    elif percentage >= 60:
        status = "🟠 NEEDS WORK"
        recommendation = "System needs significant work before publication readiness."
    else:
        status = "🔴 NOT READY"
        recommendation = "System requires major improvements before publication."
    
    return status, recommendation, percentage, all_issues

def main():
    """Run complete publication readiness check."""
    print_header()
    
    status, recommendation, score, issues = assess_publication_readiness()
    
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    print(f"Status: {status}")
    print(f"Score: {score:.1f}%")
    print(f"\nRecommendation: {recommendation}")
    
    if issues:
        print(f"\nIssues to Address ({len(issues)}):")
        for i, issue in enumerate(issues[:10], 1):  # Show first 10 issues
            print(f"  {i}. {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    
    print(f"\nAssessment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return appropriate exit code
    return 0 if score >= 75 else 1

if __name__ == "__main__":
    sys.exit(main())