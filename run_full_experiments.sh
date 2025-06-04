#!/bin/bash
# Full experimental pipeline for multi-camera classification research
# This script runs experiments, generates visualizations, and organizes results

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE_DIR="${BASE_DIR}/experimental_results"
LOG_DIR="${RESULTS_BASE_DIR}/logs"

# Create base directories
mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${LOG_DIR}"

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run experiments for a given scale
run_experiments() {
    local scale=$1
    local workers=$2
    local timeout=$3
    
    print_message $BLUE "═══════════════════════════════════════════════════════════════"
    print_message $BLUE "Running ${scale} experiments"
    print_message $BLUE "═══════════════════════════════════════════════════════════════"
    
    local output_dir="research_results_${scale}_${TIMESTAMP}"
    local log_file="${LOG_DIR}/experiment_${scale}_${TIMESTAMP}.log"
    
    print_message $YELLOW "Output directory: ${output_dir}"
    print_message $YELLOW "Log file: ${log_file}"
    print_message $YELLOW "Workers: ${workers}"
    print_message $YELLOW "Timeout: ${timeout} seconds"
    
    # Run experiments with timeout
    if timeout ${timeout} python experiments/full_scale_experiments.py \
        --scale ${scale} \
        --parallel \
        --workers ${workers} \
        --output-dir ${output_dir} \
        > "${log_file}" 2>&1; then
        
        print_message $GREEN "✓ ${scale} experiments completed successfully"
        
        # Generate visualizations
        if [ -f "${output_dir}/raw_results/all_results.csv" ]; then
            print_message $YELLOW "Generating visualizations..."
            python experiments/visualize_results.py \
                "${output_dir}/raw_results/all_results.csv" \
                "${output_dir}/figures" \
                >> "${log_file}" 2>&1
            print_message $GREEN "✓ Visualizations generated"
        else
            print_message $RED "✗ No results file found for visualization"
        fi
        
        # Move to organized structure
        local final_dir="${RESULTS_BASE_DIR}/${scale}_scale"
        mkdir -p "${final_dir}"
        mv "${output_dir}" "${final_dir}/${TIMESTAMP}"
        
        print_message $GREEN "✓ Results moved to: ${final_dir}/${TIMESTAMP}"
        
    else
        print_message $RED "✗ ${scale} experiments failed or timed out"
        print_message $YELLOW "Check log file: ${log_file}"
        return 1
    fi
}

# Function to generate combined report
generate_combined_report() {
    print_message $BLUE "═══════════════════════════════════════════════════════════════"
    print_message $BLUE "Generating combined report"
    print_message $BLUE "═══════════════════════════════════════════════════════════════"
    
    local report_file="${RESULTS_BASE_DIR}/COMBINED_RESULTS_${TIMESTAMP}.md"
    
    cat > "${report_file}" << EOF
# Multi-Camera Classification: Combined Experimental Results
Generated: $(date)

## Experiments Run

EOF
    
    # Add results from each scale
    for scale in small_scale medium_scale full_scale; do
        local scale_dir="${RESULTS_BASE_DIR}/${scale}"
        if [ -d "${scale_dir}" ]; then
            echo "### ${scale} Results" >> "${report_file}"
            echo "" >> "${report_file}"
            
            # Find most recent results
            local latest_dir=$(ls -t "${scale_dir}" | head -1)
            if [ -n "${latest_dir}" ] && [ -f "${scale_dir}/${latest_dir}/summary_report.txt" ]; then
                cat "${scale_dir}/${latest_dir}/summary_report.txt" >> "${report_file}"
                echo "" >> "${report_file}"
            else
                echo "No results found for ${scale}" >> "${report_file}"
                echo "" >> "${report_file}"
            fi
        fi
    done
    
    print_message $GREEN "✓ Combined report generated: ${report_file}"
}

# Main execution
main() {
    print_message $GREEN "╔═══════════════════════════════════════════════════════════════╗"
    print_message $GREEN "║     Multi-Camera Classification Experimental Pipeline          ║"
    print_message $GREEN "╚═══════════════════════════════════════════════════════════════╝"
    
    # Check Python environment
    print_message $YELLOW "Checking Python environment..."
    python --version
    
    # Parse command line arguments
    SCALE="${1:-all}"
    
    case $SCALE in
        small|small_scale)
            run_experiments "small_scale" 4 600
            ;;
        medium|medium_scale)
            run_experiments "medium_scale" 8 7200
            ;;
        large|full|full_scale)
            run_experiments "full_scale" 16 28800
            ;;
        all)
            # Run all scales sequentially
            run_experiments "small_scale" 4 600
            
            print_message $YELLOW "Continue with medium scale? (y/n)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                run_experiments "medium_scale" 8 7200
            fi
            
            print_message $YELLOW "Continue with full scale? (y/n)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                run_experiments "full_scale" 16 28800
            fi
            ;;
        *)
            print_message $RED "Invalid scale: $SCALE"
            print_message $YELLOW "Usage: $0 [small|medium|large|all]"
            exit 1
            ;;
    esac
    
    # Generate combined report
    generate_combined_report
    
    # Summary
    print_message $GREEN "╔═══════════════════════════════════════════════════════════════╗"
    print_message $GREEN "║                    Pipeline Complete!                          ║"
    print_message $GREEN "╚═══════════════════════════════════════════════════════════════╝"
    print_message $BLUE "Results organized in: ${RESULTS_BASE_DIR}"
    print_message $BLUE "Logs available in: ${LOG_DIR}"
    
    # Show directory structure
    print_message $YELLOW "\nResults directory structure:"
    tree -L 3 "${RESULTS_BASE_DIR}" 2>/dev/null || ls -la "${RESULTS_BASE_DIR}"
}

# Run main function
main "$@"