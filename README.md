# Stablecoin Policy Research

A comprehensive research framework for analyzing stablecoin market behavior around monetary policy events.

## Overview

This project analyzes how stablecoin markets respond to Federal Reserve policy announcements, FOMC minutes, and other monetary policy events. It combines traditional financial analysis (event studies, GARCH models) with modern NLP techniques for sentiment analysis of policy communications.

## Features

- **Data Collection**: Automated data fetching from CoinGecko, Federal Reserve, and macro data sources
- **Sentiment Analysis**: DeepSeek LLM-based sentiment analysis of FOMC minutes and speeches (cheaper and more stable than OpenAI)
- **Event Studies**: Cumulative abnormal returns and buy-and-hold abnormal returns analysis
- **Volatility Modeling**: GARCH/EGARCH models for volatility dynamics
- **Policy Transmission**: VAR models with impulse response functions

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd stablecoin-policy-research

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run quickstart
make quickstart
```

## Project Structure

```
stablecoin-policy-research/
├─ README.md                    # This file
├─ pyproject.toml              # Package configuration
├─ requirements.txt            # Pinned dependencies
├─ Makefile                    # One-command workflows
├─ .gitignore                  # Git ignore rules
├─ .env.example               # API keys template
├─ configs/                   # Configuration files
├─ data/                      # Local data cache
├─ notebooks/                  # Jupyter notebooks
├─ src/                       # Source code
├─ scripts/                   # Shell scripts
├─ tests/                     # Unit tests
└─ .github/workflows/         # CI/CD workflows
```

## Configuration

Edit `configs/config.yaml` to specify:
- Data sources and tickers
- Analysis parameters
- Output paths

## Data Sources

- **CoinGecko**: Stablecoin prices and volumes
- **Federal Reserve**: FOMC minutes, speeches, policy rates
- **FRED**: Macroeconomic indicators
- **On-chain**: Optional blockchain data via Etherscan/Dune

## Analysis Pipeline

1. **Data Ingestion**: `make ingest`
2. **Feature Engineering**: `make features`
3. **Analysis**: `make analysis`
4. **Visualization**: `make figures`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
