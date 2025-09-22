#!/usr/bin/env python3
"""
Comprehensive guide to policy impact quantification and DeepSeek API usage.
Demonstrates how to measure and quantify policy effects on stablecoins.
"""

import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.features.sentiment.llm_adapter import LLMAdapter

def explain_deepseek_api_workflow():
    """Explain how the DeepSeek API works for policy analysis."""
    
    print("🧠 DeepSeek API Workflow for Policy Analysis")
    print("=" * 60)
    
    print("""
📡 API ARCHITECTURE:

1. INPUT PREPARATION:
   - Policy text (FOMC minutes, speeches, press releases)
   - Context specification (Federal Reserve policy)
   - Text preprocessing and cleaning

2. API REQUEST STRUCTURE:
   - Endpoint: https://api.deepseek.com/v1/chat/completions
   - Method: POST
   - Headers: Authorization, Content-Type
   - Payload: Model, messages, temperature, max_tokens

3. PROMPT ENGINEERING:
   - System prompt: Define the AI's role as financial analyst
   - User prompt: Specific policy text with analysis instructions
   - Output format: Structured JSON response

4. RESPONSE PROCESSING:
   - Parse JSON response
   - Extract sentiment classification
   - Validate confidence scores
   - Map to numerical values

5. QUANTITATIVE MAPPING:
   - Sentiment → Numerical scores
   - Confidence → Weighting factors
   - Time series → Market alignment
""")

def demonstrate_api_call():
    """Demonstrate actual DeepSeek API call."""
    
    print("\n🔧 DeepSeek API Call Demonstration")
    print("=" * 50)
    
    # Sample policy text
    policy_text = """
    The Federal Reserve remains committed to achieving maximum employment and price stability. 
    Given the current economic conditions and inflationary pressures, we believe it is appropriate 
    to maintain our current accommodative stance while monitoring incoming data carefully. 
    The Committee will continue to assess additional information and its implications for monetary policy.
    """
    
    print("📝 Sample Policy Text:")
    print(f"'{policy_text.strip()}'")
    
    # Initialize LLM adapter
    api_key = "sk-8990403e972a4624bb313314927bc4c2"
    llm_adapter = LLMAdapter(api_key=api_key, model="deepseek-chat")
    
    print("\n🔄 Making API Call...")
    
    try:
        # Analyze sentiment
        result = llm_adapter.analyze_sentiment(policy_text, "Federal Reserve monetary policy")
        
        print("✅ API Response Received:")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Key Phrases: {result['key_phrases']}")
        print(f"   Explanation: {result['explanation']}")
        
        # Show quantitative mapping
        sentiment_scores = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
        numerical_score = sentiment_scores.get(result['sentiment'], 0)
        weighted_score = numerical_score * result['confidence']
        
        print(f"\n📊 Quantitative Mapping:")
        print(f"   Categorical: {result['sentiment']}")
        print(f"   Numerical: {numerical_score}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Weighted Score: {weighted_score:.3f}")
        
    except Exception as e:
        print(f"❌ API Call Failed: {e}")

def explain_policy_quantification_methods():
    """Explain different methods for quantifying policy impact."""
    
    print("\n📊 Policy Impact Quantification Methods")
    print("=" * 50)
    
    print("""
🎯 METHOD 1: SENTIMENT SCORING

A. Categorical Classification:
   - Hawkish: +1 (tightening policy signals)
   - Neutral: 0 (balanced or data-dependent)
   - Dovish: -1 (accommodative policy signals)

B. Confidence Weighting:
   - Raw Confidence: 0.0 to 1.0 scale
   - Weighted Score: sentiment_score × confidence
   - Magnitude: |sentiment_score| × confidence

C. Time Series Creation:
   - Map policy events to market dates
   - Create continuous sentiment series
   - Handle missing data and alignment

📈 METHOD 2: EVENT STUDY ANALYSIS

A. Event Identification:
   - Policy announcement dates
   - Market trading days alignment
   - Event window definition (±5 days)

B. Abnormal Performance Calculation:
   - Estimation Window: 250 days before event
   - Normal Performance: Expected returns/volatility
   - Abnormal Performance: Actual - Expected

C. Statistical Testing:
   - t-tests: Significance of abnormal performance
   - Cumulative Abnormal Returns (CAR)
   - Buy-and-Hold Abnormal Returns (BHAR)

🔍 METHOD 3: REGRESSION ANALYSIS

A. Dependent Variables:
   - Stablecoin returns
   - Peg deviations
   - Volatility measures
   - Trading volume

B. Independent Variables:
   - Policy sentiment scores
   - Market control variables
   - Time dummies
   - Interaction terms

C. Model Specifications:
   - OLS: Linear relationships
   - GARCH: Volatility modeling
   - VAR: Dynamic relationships
   - Panel: Cross-stablecoin analysis

📊 METHOD 4: MACHINE LEARNING APPROACHES

A. Feature Engineering:
   - Sentiment scores
   - Market indicators
   - Technical indicators
   - Macro variables

B. Prediction Models:
   - Random Forest: Non-linear relationships
   - LSTM: Time series patterns
   - XGBoost: Gradient boosting
   - Ensemble: Multiple model combination

C. Validation:
   - Time series cross-validation
   - Out-of-sample testing
   - Robustness checks
""")

def demonstrate_policy_impact_calculation():
    """Demonstrate how to calculate policy impact on stablecoins."""
    
    print("\n🧮 Policy Impact Calculation Example")
    print("=" * 50)
    
    # Load sample data
    try:
        prices_df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        peg_deviations_df = pd.read_parquet('data/processed/peg_deviations.parquet')
        sentiment_df = pd.read_csv('policy_sentiment_results.csv')
        
        print("✅ Data loaded successfully")
        
        # Example: Calculate policy impact for USDT
        stablecoin = 'USDT'
        
        print(f"\n📊 Policy Impact Analysis for {stablecoin}")
        print("-" * 40)
        
        # Get price data
        prices = prices_df[stablecoin]
        peg_deviations = peg_deviations_df[stablecoin]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Get sentiment data
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Map sentiment to market dates
        market_dates = prices.index
        sentiment_scores = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
        
        # Create sentiment time series
        sentiment_ts = pd.Series(0.0, index=market_dates)
        for _, event in sentiment_df.iterrows():
            event_date = event['date']
            sentiment_score = sentiment_scores.get(event['sentiment'], 0)
            confidence = event['confidence']
            
            # Find closest market date
            closest_date_idx = market_dates.get_indexer([event_date], method='nearest')[0]
            closest_date = market_dates[closest_date_idx]
            
            sentiment_ts.loc[closest_date] = sentiment_score * confidence
        
        # Calculate correlations
        price_correlation = prices.corr(sentiment_ts)
        return_correlation = returns.corr(sentiment_ts)
        peg_correlation = peg_deviations.corr(sentiment_ts)
        
        print(f"Price Correlation: {price_correlation:.4f}")
        print(f"Return Correlation: {return_correlation:.4f}")
        print(f"Peg Deviation Correlation: {peg_correlation:.4f}")
        
        # Event study example
        print(f"\n🎯 Event Study Analysis:")
        
        # Find policy event dates
        policy_dates = sentiment_ts[sentiment_ts != 0].index
        
        for i, event_date in enumerate(policy_dates):
            print(f"\nEvent {i+1}: {event_date.date()}")
            
            # Define event window
            pre_window = 5
            post_window = 5
            
            # Get event window data
            event_start = event_date - pd.Timedelta(days=pre_window)
            event_end = event_date + pd.Timedelta(days=post_window)
            
            event_prices = prices[(prices.index >= event_start) & (prices.index <= event_end)]
            event_returns = returns[(returns.index >= event_start) & (returns.index <= event_end)]
            event_peg = peg_deviations[(peg_deviations.index >= event_start) & (peg_deviations.index <= event_end)]
            
            if len(event_prices) > 0:
                # Calculate abnormal performance
                normal_return = returns.mean()
                event_abnormal_returns = event_returns - normal_return
                cumulative_abnormal_return = event_abnormal_returns.sum()
                
                # Calculate peg stability
                avg_peg_deviation = abs(event_peg).mean()
                peg_stability = 1 - avg_peg_deviation
                
                print(f"   Cumulative Abnormal Return: {cumulative_abnormal_return:.6f}")
                print(f"   Average Peg Deviation: {avg_peg_deviation:.6f}")
                print(f"   Peg Stability: {peg_stability:.6f}")
                
                # Statistical significance
                if len(event_abnormal_returns) > 1:
                    t_stat = event_abnormal_returns.mean() / event_abnormal_returns.std() * np.sqrt(len(event_abnormal_returns))
                    print(f"   t-statistic: {t_stat:.3f}")
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Total Policy Events: {len(policy_dates)}")
        print(f"   Average Price: ${prices.mean():.6f}")
        print(f"   Average Peg Deviation: {abs(peg_deviations).mean():.6f}")
        print(f"   Price Volatility: {returns.std():.6f}")
        
    except Exception as e:
        print(f"❌ Error in calculation: {e}")

def explain_advanced_quantification_methods():
    """Explain advanced methods for policy quantification."""
    
    print("\n🔬 Advanced Policy Quantification Methods")
    print("=" * 50)
    
    print("""
🎯 METHOD 1: HIGH-FREQUENCY ANALYSIS

A. Intraday Data:
   - Minute-by-minute price data
   - Real-time sentiment analysis
   - Immediate market reactions
   - Microstructure effects

B. Event Time Analysis:
   - Pre-event: 1 hour before announcement
   - Event: Announcement time
   - Post-event: 1 hour after announcement
   - Recovery: Return to normal levels

C. Market Microstructure:
   - Bid-ask spreads
   - Order book depth
   - Trading volume spikes
   - Liquidity provision

📊 METHOD 2: CROSS-ASSET ANALYSIS

A. Asset Classes:
   - Stablecoins (USDT, USDC, DAI)
   - Cryptocurrencies (BTC, ETH)
   - Traditional assets (USD, Gold, Bonds)
   - Equity indices (S&P 500, NASDAQ)

B. Correlation Analysis:
   - Dynamic correlations
   - Rolling correlation windows
   - Regime-dependent correlations
   - Causality testing

C. Portfolio Effects:
   - Diversification benefits
   - Risk transmission
   - Systemic risk measures
   - Contagion analysis

🔍 METHOD 3: MACHINE LEARNING APPROACHES

A. Natural Language Processing:
   - BERT-based sentiment analysis
   - Transformer models
   - Attention mechanisms
   - Contextual embeddings

B. Time Series Forecasting:
   - LSTM networks
   - GRU models
   - Transformer architectures
   - Ensemble methods

C. Reinforcement Learning:
   - Policy optimization
   - Market making strategies
   - Arbitrage detection
   - Risk management

📈 METHOD 4: CAUSAL INFERENCE

A. Instrumental Variables:
   - Policy surprises as instruments
   - Exogenous variation
   - Endogeneity correction
   - Robust identification

B. Difference-in-Differences:
   - Treated vs control groups
   - Policy implementation timing
   - Parallel trends assumption
   - Robustness checks

C. Regression Discontinuity:
   - Policy thresholds
   - Sharp discontinuities
   - Local linear regression
   - Bandwidth selection

🎯 METHOD 5: REGIME-SWITCHING MODELS

A. Markov Switching:
   - High/low volatility regimes
   - Policy impact by regime
   - Transition probabilities
   - Duration analysis

B. Threshold Models:
   - Policy threshold effects
   - Asymmetric responses
   - Non-linear relationships
   - Breakpoint detection

C. GARCH Models:
   - Volatility clustering
   - Policy impact on volatility
   - Leverage effects
   - Long memory processes
""")

def create_policy_impact_dashboard():
    """Create a comprehensive policy impact dashboard."""
    
    print("\n📊 Creating Policy Impact Dashboard")
    print("=" * 50)
    
    try:
        # Load data
        prices_df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        peg_deviations_df = pd.read_parquet('data/processed/peg_deviations.parquet')
        sentiment_df = pd.read_csv('policy_sentiment_results.csv')
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Policy Impact Quantification Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution
        ax1 = axes[0, 0]
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        colors = ['red' if s == 'hawkish' else 'blue' if s == 'dovish' else 'gray' for s in sentiment_counts.index]
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Policy Sentiment Distribution')
        ax1.set_ylabel('Number of Events')
        
        # Add value labels
        for bar, value in zip(bars, sentiment_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence Distribution
        ax2 = axes[0, 1]
        ax2.hist(sentiment_df['confidence'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Sentiment Confidence Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.axvline(sentiment_df['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {sentiment_df["confidence"].mean():.3f}')
        ax2.legend()
        
        # 3. Stablecoin Peg Stability
        ax3 = axes[0, 2]
        stablecoins = ['USDT', 'USDC', 'DAI', 'FRAX', 'LUSD', 'TUSD', 'USDP', 'BUSD']
        peg_stability = []
        
        for coin in stablecoins:
            if coin in peg_deviations_df.columns:
                avg_abs_dev = abs(peg_deviations_df[coin]).mean()
                stability = 1 - avg_abs_dev
                peg_stability.append(stability)
            else:
                peg_stability.append(0)
        
        bars = ax3.bar(stablecoins, peg_stability, color='lightgreen', alpha=0.7)
        ax3.set_title('Stablecoin Peg Stability')
        ax3.set_ylabel('Peg Stability (1 - Avg Abs Deviation)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0.99, 1.0)
        
        # Add value labels
        for bar, value in zip(bars, peg_stability):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 4. Price Evolution
        ax4 = axes[1, 0]
        for coin in ['USDT', 'USDC', 'DAI']:
            if coin in prices_df.columns:
                ax4.plot(prices_df.index, prices_df[coin], label=coin, alpha=0.7)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Peg')
        ax4.set_title('Stablecoin Price Evolution')
        ax4.set_ylabel('Price (USD)')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Peg Deviations Over Time
        ax5 = axes[1, 1]
        for coin in ['USDT', 'USDC', 'DAI']:
            if coin in peg_deviations_df.columns:
                ax5.plot(peg_deviations_df.index, abs(peg_deviations_df[coin]), label=coin, alpha=0.7)
        ax5.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Stress Threshold (1%)')
        ax5.set_title('Absolute Peg Deviations Over Time')
        ax5.set_ylabel('Absolute Deviation from $1.00')
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Policy Event Timeline
        ax6 = axes[1, 2]
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_scores = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
        
        for _, event in sentiment_df.iterrows():
            color = 'red' if event['sentiment'] == 'hawkish' else 'blue' if event['sentiment'] == 'dovish' else 'gray'
            ax6.scatter(event['date'], sentiment_scores[event['sentiment']], 
                       color=color, s=100, alpha=0.7)
        
        ax6.set_title('Policy Events Timeline')
        ax6.set_ylabel('Sentiment Score')
        ax6.set_xlabel('Date')
        ax6.tick_params(axis='x', rotation=45)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/policy_impact_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dashboard saved to data/processed/policy_impact_dashboard.png")
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")

def main():
    """Main function to demonstrate policy quantification."""
    
    print("🔬 Policy Impact Quantification & DeepSeek API Guide")
    print("=" * 70)
    
    # Explain API workflow
    explain_deepseek_api_workflow()
    
    # Demonstrate API call
    demonstrate_api_call()
    
    # Explain quantification methods
    explain_policy_quantification_methods()
    
    # Demonstrate calculations
    demonstrate_policy_impact_calculation()
    
    # Explain advanced methods
    explain_advanced_quantification_methods()
    
    # Create dashboard
    create_policy_impact_dashboard()
    
    print("\n🎉 Policy Quantification Guide Complete!")
    print("\n💡 Key Takeaways:")
    print("   1. DeepSeek API provides reliable sentiment classification")
    print("   2. Multiple quantification methods available")
    print("   3. Event study analysis reveals policy impact")
    print("   4. Machine learning approaches enhance accuracy")
    print("   5. Causal inference methods ensure robust results")

if __name__ == "__main__":
    main()
