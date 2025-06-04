#!/usr/bin/env python3
"""Check if the environment is ready for running experiments."""

import sys
import os
from pathlib import Path

def check_environment():
    """Check if all requirements are met."""
    print("Multi-Camera Classification Experiment Setup Checker")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Check Python version
    print("\n1. Checking Python version...")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 7:
        print(f"   ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        issues.append(f"Python 3.7+ required, found {py_version.major}.{py_version.minor}")
    
    # Check required packages
    print("\n2. Checking required packages...")
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml'
    }
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✓ {pip_name}")
        except ImportError:
            issues.append(f"Missing package: {pip_name}")
    
    # Check directory structure
    print("\n3. Checking directory structure...")
    required_dirs = [
        'src/algorithms',
        'src/algorithms/baselines',
        'src/core',
        'src/game_theory',
        'src/visualization',
        'experiments',
        'configs'
    ]
    
    base_dir = Path(__file__).parent
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"   ✓ {dir_path}")
        else:
            issues.append(f"Missing directory: {dir_path}")
    
    # Check key files
    print("\n4. Checking key files...")
    key_files = [
        'experiments/full_scale_experiments.py',
        'experiments/visualize_results.py',
        'src/main.py',
        'run_full_experiments.sh'
    ]
    
    for file_path in key_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            issues.append(f"Missing file: {file_path}")
    
    # Check algorithms
    print("\n5. Checking algorithm implementations...")
    algorithm_files = [
        'src/algorithms/fixed_frequency.py',
        'src/algorithms/variable_frequency.py',
        'src/algorithms/unknown_frequency.py',
        'src/algorithms/baselines/random_selection.py',
        'src/algorithms/baselines/greedy_energy.py'
    ]
    
    for algo_file in algorithm_files:
        full_path = base_dir / algo_file
        if full_path.exists():
            print(f"   ✓ {algo_file}")
        else:
            warnings.append(f"Missing algorithm: {algo_file}")
    
    # Check disk space
    print("\n6. Checking disk space...")
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    if free_gb > 10:
        print(f"   ✓ {free_gb} GB free")
    else:
        warnings.append(f"Low disk space: {free_gb} GB free (10+ GB recommended)")
    
    # Check CPU cores
    print("\n7. Checking CPU cores...")
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        print(f"   ✓ {cores} CPU cores available")
        if cores < 4:
            warnings.append(f"Only {cores} CPU cores (4+ recommended for parallel execution)")
    except:
        warnings.append("Could not determine CPU count")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not issues and not warnings:
        print("\n✅ All checks passed! Ready to run experiments.")
        print("\nTo run experiments:")
        print("  ./run_full_experiments.sh small    # Quick test")
        print("  ./run_full_experiments.sh medium   # Full evaluation")
        print("  ./run_full_experiments.sh large    # Comprehensive")
        return True
    else:
        if issues:
            print("\n❌ Critical issues found:")
            for issue in issues:
                print(f"   - {issue}")
        
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("\nTo fix issues:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Ensure you're in the correct directory")
        print("  3. Check that all files were properly downloaded")
        return False

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)