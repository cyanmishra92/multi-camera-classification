#!/usr/bin/env python3
"""
Master test runner for all scenario tests.

Runs all test scenarios and generates a combined report.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

# Test scripts to run
TEST_SCRIPTS = [
    "run_comprehensive_test.py",
    "test_edge_cases.py",
    "test_stress.py",
    "test_parameter_sensitivity.py",
    "test_algorithm_comparison.py"
]

TEST_DESCRIPTIONS = {
    "run_comprehensive_test.py": "Comprehensive system test with visualizations",
    "test_edge_cases.py": "Edge case and boundary condition testing",
    "test_stress.py": "Stress testing and scalability analysis",
    "test_parameter_sensitivity.py": "Parameter sensitivity analysis",
    "test_algorithm_comparison.py": "Algorithm performance comparison"
}


def run_test_script(script_name, args=[]):
    """Run a single test script."""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"Description: {TEST_DESCRIPTIONS.get(script_name, 'No description')}")
    print(f"{'='*80}")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        # Run the script
        cmd = [sys.executable, str(script_path)] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 3:
                print("Output summary:")
                for line in output_lines[-3:]:
                    print(f"  {line}")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {str(e)}")
        return False


def generate_master_report(results, output_dir):
    """Generate a master report summarizing all tests."""
    report_path = output_dir / "master_test_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MASTER TEST REPORT - MULTI-CAMERA CLASSIFICATION SYSTEM\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test summary
        f.write("TEST EXECUTION SUMMARY\n")
        f.write("-"*40 + "\n")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r)
        
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {total_tests - passed_tests}\n")
        f.write(f"Success rate: {passed_tests/total_tests*100:.1f}%\n\n")
        
        # Individual test results
        f.write("INDIVIDUAL TEST RESULTS\n")
        f.write("-"*40 + "\n")
        
        for script, success in results.items():
            status = "PASSED" if success else "FAILED"
            f.write(f"{script:<40} {status}\n")
            f.write(f"  {TEST_DESCRIPTIONS.get(script, '')}\n\n")
        
        # Test descriptions
        f.write("\nTEST DESCRIPTIONS\n")
        f.write("-"*40 + "\n")
        
        f.write("\n1. Comprehensive Test:\n")
        f.write("   - Runs all three algorithms with standard parameters\n")
        f.write("   - Generates energy, accuracy, and performance visualizations\n")
        f.write("   - Creates detailed performance reports\n")
        
        f.write("\n2. Edge Case Testing:\n")
        f.write("   - Tests extreme scenarios (low battery, high frequency, etc.)\n")
        f.write("   - Identifies algorithm breaking points\n")
        f.write("   - Validates robustness under adverse conditions\n")
        
        f.write("\n3. Stress Testing:\n")
        f.write("   - Tests scalability with large numbers of cameras\n")
        f.write("   - Measures performance under heavy load\n")
        f.write("   - Analyzes memory usage and execution time\n")
        
        f.write("\n4. Parameter Sensitivity:\n")
        f.write("   - Analyzes impact of parameter changes\n")
        f.write("   - Identifies most sensitive parameters\n")
        f.write("   - Provides tuning recommendations\n")
        
        f.write("\n5. Algorithm Comparison:\n")
        f.write("   - Compares algorithms across diverse scenarios\n")
        f.write("   - Identifies optimal algorithm for each use case\n")
        f.write("   - Provides selection guidelines\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        
        if passed_tests == total_tests:
            f.write("\n✓ All tests passed successfully!\n")
            f.write("  The system appears to be functioning correctly.\n")
        else:
            f.write("\n⚠ Some tests failed. Please review individual test outputs.\n")
            f.write("  Failed tests:\n")
            for script, success in results.items():
                if not success:
                    f.write(f"    - {script}\n")
        
        f.write("\nNext steps:\n")
        f.write("1. Review individual test reports in test_results_* directories\n")
        f.write("2. Analyze visualizations for performance insights\n")
        f.write("3. Use parameter sensitivity results for system tuning\n")
        f.write("4. Select appropriate algorithm based on comparison results\n")
        
        f.write("\n" + "="*80 + "\n")
        
        # Output locations
        f.write("\nOUTPUT LOCATIONS\n")
        f.write("-"*40 + "\n")
        f.write("Look for results in directories matching:\n")
        f.write("  - test_results_YYYYMMDD_HHMMSS/\n")
        f.write("  - test_results_edge_cases_*/\n")
        f.write("  - test_results_stress_*/\n")
        f.write("  - test_results_sensitivity_*/\n")
        f.write("  - test_results_comparison_*/\n")
        
        f.write("\n" + "="*80 + "\n")


def main():
    """Run all test scenarios."""
    parser = argparse.ArgumentParser(
        description="Run all test scenarios for multi-camera classification system"
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        help='Skip specific tests (e.g., --skip stress sensitivity)',
        default=[]
    )
    parser.add_argument(
        '--only',
        nargs='+',
        help='Run only specific tests (e.g., --only comprehensive edge)',
        default=[]
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    tests_to_run = TEST_SCRIPTS.copy()
    
    if args.only:
        # Filter to only requested tests
        tests_to_run = []
        for test in TEST_SCRIPTS:
            for keyword in args.only:
                if keyword.lower() in test.lower():
                    tests_to_run.append(test)
                    break
    
    if args.skip:
        # Remove skipped tests
        for test in TEST_SCRIPTS:
            for keyword in args.skip:
                if keyword.lower() in test.lower() and test in tests_to_run:
                    tests_to_run.remove(test)
                    break
    
    print("="*80)
    print("MULTI-CAMERA CLASSIFICATION - MASTER TEST RUNNER")
    print("="*80)
    print(f"Running {len(tests_to_run)} tests...")
    
    # Create output directory for master report
    output_dir = Path(f"test_results_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    
    # Run tests
    results = {}
    for test_script in tests_to_run:
        success = run_test_script(test_script)
        results[test_script] = success
    
    # Generate master report
    print("\nGenerating master report...")
    generate_master_report(results, output_dir)
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\n{'='*80}")
    print("TEST EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"\nMaster report saved to: {output_dir}/master_test_report.txt")
    print(f"{'='*80}")
    
    # Return non-zero exit code if any tests failed
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()