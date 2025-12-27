#!/bin/bash
# Convenience script to run hyperparameter search
# Usage: bash scripts/run_hyperparameter_search.sh

echo "=========================================="
echo "Hyperparameter Search"
echo "=========================================="
echo ""
echo "This will run training + evaluation for all experiments."
echo "Estimated time: 6-8 hours (depending on GPU)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run the search
python scripts/hyperparameter_search.py --skip-completed

echo ""
echo "=========================================="
echo "Search Complete!"
echo "=========================================="
echo ""
echo "Results saved to: outputs/hyperparameter_search_results.json"
echo ""
echo "To view results:"
echo "  python scripts/hyperparameter_search.py --compare"
echo ""

