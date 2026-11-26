#!/usr/bin/env bash
# Placeholder script – replace with the actual download commands
# once the dataset is publicly available or you have the Kaggle API key.

set -euo pipefail

echo "=== Fundus-CIMT data download helper ==="
echo ""
echo "1. Install Kaggle CLI:  pip install kaggle"
echo "2. Place kaggle.json in ~/.kaggle/"
echo "3. Then run something like:"
echo ""
echo "   kaggle datasets download -d rofikbme/fundus-image -p data/raw --unzip"
echo "   kaggle datasets download -d amimulhasan/fundus-imagel -p data/raw --unzip"
echo ""
echo "After download, update paths in config/config.yaml if necessary."
echo "Done."
