#!/bin/bash

# Script to generate figures and visualizations
# This script runs the figure generation notebook

set -e  # Exit on any error

echo "📊 Generating Figures for Stablecoin Policy Research"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if jupyter is available
if ! command -v jupyter &> /dev/null; then
    echo "❌ Jupyter is not installed. Installing..."
    pip install jupyter
fi

# Check if required data exists
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed)" ]; then
    echo "❌ No processed data found. Please run feature engineering first:"
    echo "   python -m src.pipelines.run_features"
    exit 1
fi

# Create figures directory
mkdir -p figures

# Run the figures notebook
echo "📈 Running figure generation notebook..."
jupyter nbconvert --execute --to notebook --inplace notebooks/10_figures.ipynb

# Convert to HTML for easy viewing
echo "🌐 Converting notebook to HTML..."
jupyter nbconvert --to html notebooks/10_figures.ipynb --output-dir=figures

echo "✅ Figures generated successfully!"
echo ""
echo "📁 Check the following locations:"
echo "   - figures/ directory for generated plots"
echo "   - figures/10_figures.html for interactive notebook"
echo "   - notebooks/10_figures.ipynb for the executed notebook"
echo ""
echo "🎨 You can also open notebooks/10_figures.ipynb in Jupyter to:"
echo "   - Modify figure parameters"
echo "   - Generate additional visualizations"
echo "   - Export figures in different formats"
