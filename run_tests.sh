#!/bin/bash
# Launcher script for comprehensive tests

echo "Multi-Camera Classification System - Test Launcher"
echo "=================================================="
echo ""

# Default values
DURATION=1000
FREQUENCY=0.1
CONFIG="configs/default_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -f|--frequency)
            FREQUENCY="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -d, --duration <value>    Simulation duration (default: 1000)"
            echo "  -f, --frequency <value>   Classification frequency (default: 0.1)"
            echo "  -c, --config <path>       Config file path (default: configs/default_config.yaml)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running comprehensive test with:"
echo "  Duration: $DURATION"
echo "  Frequency: $FREQUENCY"
echo "  Config: $CONFIG"
echo ""

# Run the comprehensive test
python tests/scenarios/run_comprehensive_test.py \
    --duration "$DURATION" \
    --frequency "$FREQUENCY" \
    --config "$CONFIG"

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Test completed successfully!"
    echo "Check the test_results_* directory for outputs."
else
    echo ""
    echo "Test failed. Please check the error messages above."
    exit 1
fi