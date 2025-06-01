#!/usr/bin/env python3
"""Run full demonstration with all algorithms and generate comprehensive results."""

import sys
import subprocess
import time
import json
from pathlib import Path

print("=" * 80)
print("MULTI-CAMERA CLASSIFICATION SYSTEM - FULL DEMONSTRATION")
print("=" * 80)
print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# List of demo scripts to run
demo_scripts = [
    {
        'name': 'Basic Simulation',
        'script': 'run_simulation.py',
        'args': ['--duration', '2000', '--algorithm', 'fixed'],
        'output': 'demo_basic_results.json'
    },
    {
        'name': 'Algorithm Comparison',
        'script': 'examples/compare_algorithms.py',
        'args': [],
        'output': None  # Creates its own output files
    },
    {
        'name': 'Enhanced Accuracy Test',
        'script': 'test_enhanced_accuracy.py',
        'args': [],
        'output': None
    },
    {
        'name': 'Game Theory Test',
        'script': 'test_game_theory.py', 
        'args': [],
        'output': None
    },
    {
        'name': 'Adaptive Algorithm Test',
        'script': 'test_adaptive_algorithm.py',
        'args': [],
        'output': None
    }
]

# Results collection
all_results = {
    'demo_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'demo_results': {}
}

# Run each demo
for demo in demo_scripts:
    print(f"\n{'='*60}")
    print(f"Running: {demo['name']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Build command
        cmd = [sys.executable, demo['script']] + demo['args']
        if demo['output']:
            cmd.extend(['--output', demo['output']])
        
        # Run the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ {demo['name']} completed successfully in {elapsed:.1f}s")
            
            # Extract key results from output
            output_lines = result.stdout.strip().split('\n')
            
            # Look for accuracy results
            for line in output_lines[-20:]:  # Check last 20 lines
                if 'Accuracy:' in line and not 'Recent' in line:
                    print(f"  {line.strip()}")
                elif 'Recent Accuracy:' in line:
                    print(f"  {line.strip()}")
                elif 'Cameras/Classification:' in line:
                    print(f"  {line.strip()}")
            
            all_results['demo_results'][demo['name']] = {
                'status': 'success',
                'runtime': elapsed,
                'output_summary': output_lines[-10:]  # Last 10 lines
            }
        else:
            print(f"✗ {demo['name']} failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:200]}...")
            
            all_results['demo_results'][demo['name']] = {
                'status': 'failed',
                'runtime': elapsed,
                'error': result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ {demo['name']} timed out after 5 minutes")
        all_results['demo_results'][demo['name']] = {
            'status': 'timeout',
            'runtime': 300
        }
    except Exception as e:
        print(f"✗ {demo['name']} failed with error: {str(e)}")
        all_results['demo_results'][demo['name']] = {
            'status': 'error',
            'error': str(e)
        }

# Generate comprehensive visualization report
print(f"\n{'='*60}")
print("Generating Comprehensive Visualization Report")
print(f"{'='*60}")

try:
    from src.visualization.comprehensive_plots import ComprehensiveVisualizer
    
    visualizer = ComprehensiveVisualizer()
    
    # Find the most recent results file
    if Path('results.json').exists():
        print("Creating comprehensive analysis from results.json...")
        visualizer.create_full_analysis_report('results.json', 'DEMO_comprehensive_analysis.pdf')
        print("✓ Comprehensive report saved to: DEMO_comprehensive_analysis.pdf")
    
    # Create comparison report if multiple algorithm results exist
    comparison_files = []
    for f in ['results_fixed_comparison.json', 'results_variable_comparison.json', 
              'results_unknown_comparison.json']:
        if Path(f).exists():
            comparison_files.append(f)
    
    if len(comparison_files) > 1:
        print(f"\nCreating algorithm comparison report from {len(comparison_files)} files...")
        from src.visualization.comprehensive_plots import create_comparison_report
        create_comparison_report(comparison_files, 'DEMO_algorithm_comparison.pdf')
        print("✓ Comparison report saved to: DEMO_algorithm_comparison.pdf")
        
except Exception as e:
    print(f"✗ Visualization generation failed: {str(e)}")
    all_results['visualization_status'] = 'failed'
else:
    all_results['visualization_status'] = 'success'

# Collect all generated files
print(f"\n{'='*60}")
print("Collecting Generated Files")
print(f"{'='*60}")

generated_files = {
    'PDF Reports': [],
    'JSON Results': [],
    'Other Files': []
}

# Check for PDF files
for pdf_file in Path('.').glob('*.pdf'):
    generated_files['PDF Reports'].append(pdf_file.name)
    print(f"  📊 {pdf_file.name}")

# Check for JSON results
for json_file in Path('.').glob('*results*.json'):
    generated_files['JSON Results'].append(json_file.name)
    print(f"  📁 {json_file.name}")

# Save demo summary
all_results['generated_files'] = generated_files
all_results['summary'] = {
    'total_demos_run': len(demo_scripts),
    'successful_demos': sum(1 for d in all_results['demo_results'].values() if d['status'] == 'success'),
    'total_runtime': sum(d.get('runtime', 0) for d in all_results['demo_results'].values())
}

with open('DEMO_SUMMARY.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("DEMONSTRATION COMPLETE")
print(f"{'='*60}")
print(f"Total runtime: {all_results['summary']['total_runtime']:.1f} seconds")
print(f"Successful demos: {all_results['summary']['successful_demos']}/{all_results['summary']['total_demos_run']}")
print(f"\nGenerated files:")
print(f"  - {len(generated_files['PDF Reports'])} PDF reports")
print(f"  - {len(generated_files['JSON Results'])} JSON result files")
print(f"\nDemo summary saved to: DEMO_SUMMARY.json")
print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)