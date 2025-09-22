#!/bin/bash

# Quickstart script for stablecoin policy research
# This script runs the complete pipeline from data ingestion to analysis

set -e  # Exit on any error

echo "🚀 Starting Stablecoin Policy Research Quickstart"
echo "================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys before continuing"
    echo "   Required: OPENAI_API_KEY, FRED_API_KEY"
    echo "   Optional: COINGECKO_API_KEY, ETHERSCAN_API_KEY, DUNE_API_KEY"
    read -p "Press Enter after updating .env file..."
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/interim data/processed
mkdir -p outputs results figures logs

# Run data ingestion
echo "📊 Running data ingestion..."
python -m src.pipelines.run_ingest

# Run feature engineering
echo "🔧 Running feature engineering..."
python -m src.pipelines.run_features

# Run analysis
echo "📈 Running analysis..."
python -m src.pipelines.run_analysis

# Generate figures
echo "📊 Generating figures..."
python scripts/make_figures.sh

echo "✅ Quickstart completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Check results/ directory for analysis outputs"
echo "   2. Check figures/ directory for visualizations"
echo "   3. Open notebooks/00_exploration.ipynb to explore the data"
echo "   4. Open notebooks/10_figures.ipynb to generate additional figures"
echo ""
echo "🎉 Happy researching!"
