#!/usr/bin/env python3
"""
Analyze correlation between policy sentiment and stablecoin volatility.
Demonstrates quantitative research methodology for news analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.features.sentiment.llm_adapter import LLMAdapter

def load_market_data() -> pd.DataFrame:
    """Load stablecoin market data."""
    try:
        prices_df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        volatility_df = pd.read_parquet('data/processed/volatility.parquet')
        returns_df = pd.read_parquet('data/processed/returns.parquet')
        
        print(f"✅ Loaded market data:")
        print(f"   Prices: {prices_df.shape}")
        print(f"   Volatility: {volatility_df.shape}")
        print(f"   Returns: {returns_df.shape}")
        
        return prices_df, volatility_df, returns_df
    except Exception as e:
        print(f"❌ Error loading market data: {e}")
        return None, None, None

def load_policy_sentiment() -> pd.DataFrame:
    """Load policy sentiment data."""
    try:
        sentiment_df = pd.read_csv('policy_sentiment_results.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        print(f"✅ Loaded policy sentiment data: {sentiment_df.shape}")
        return sentiment_df
    except Exception as e:
        print(f"❌ Error loading sentiment data: {e}")
        return None

def create_sentiment_timeseries(sentiment_df: pd.DataFrame, market_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create sentiment time series aligned with market data."""
    
    print("\n🔄 Creating sentiment time series...")
    
    # Create sentiment score mapping
    sentiment_scores = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
    
    # Initialize time series with zeros
    sentiment_ts = pd.DataFrame(index=market_dates)
    sentiment_ts['policy_sentiment'] = 0.0
    sentiment_ts['sentiment_confidence'] = 0.0
    sentiment_ts['sentiment_magnitude'] = 0.0
    
    # Convert market dates to timezone-naive for comparison
    market_dates_naive = market_dates.tz_localize(None)
    
    # Map policy events to market dates
    for _, event in sentiment_df.iterrows():
        event_date = event['date']
        sentiment_score = sentiment_scores.get(event['sentiment'], 0)
        confidence = event['confidence']
        
        # Find closest market date
        closest_date_idx = market_dates_naive.get_indexer([event_date], method='nearest')[0]
        closest_date = market_dates[closest_date_idx]
        
        # Update sentiment for that date
        sentiment_ts.loc[closest_date, 'policy_sentiment'] = sentiment_score
        sentiment_ts.loc[closest_date, 'sentiment_confidence'] = confidence
        sentiment_ts.loc[closest_date, 'sentiment_magnitude'] = abs(sentiment_score) * confidence
    
    print(f"✅ Created sentiment time series: {sentiment_ts.shape}")
    print(f"   Non-zero sentiment days: {(sentiment_ts['policy_sentiment'] != 0).sum()}")
    
    return sentiment_ts

def calculate_sentiment_volatility_correlation(volatility_df: pd.DataFrame, sentiment_ts: pd.DataFrame) -> Dict:
    """Calculate correlation between sentiment and volatility."""
    
    print("\n📊 Calculating sentiment-volatility correlations...")
    
    correlations = {}
    
    for stablecoin in volatility_df.columns:
        if stablecoin in ['timestamp']:
            continue
            
        # Align volatility and sentiment data
        vol_data = volatility_df[stablecoin]
        sent_data = sentiment_ts['policy_sentiment']
        
        # Calculate correlations
        correlation = vol_data.corr(sent_data)
        
        # Calculate rolling correlation (30-day window)
        rolling_corr = vol_data.rolling(window=30).corr(sent_data)
        
        # Calculate correlation during policy events only
        policy_days = sentiment_ts['policy_sentiment'] != 0
        if policy_days.sum() > 0:
            policy_correlation = vol_data[policy_days].corr(sent_data[policy_days])
        else:
            policy_correlation = np.nan
        
        correlations[stablecoin] = {
            'overall_correlation': correlation,
            'policy_event_correlation': policy_correlation,
            'rolling_correlation_mean': rolling_corr.mean(),
            'rolling_correlation_std': rolling_corr.std()
        }
        
        print(f"   {stablecoin}: Overall={correlation:.3f}, Policy Events={policy_correlation:.3f}")
    
    return correlations

def analyze_sentiment_impact_on_volatility(volatility_df: pd.DataFrame, sentiment_ts: pd.DataFrame) -> Dict:
    """Analyze how sentiment impacts volatility."""
    
    print("\n🎯 Analyzing sentiment impact on volatility...")
    
    results = {}
    
    for stablecoin in volatility_df.columns:
        if stablecoin in ['timestamp']:
            continue
            
        vol_data = volatility_df[stablecoin]
        
        # Separate volatility by sentiment
        hawkish_days = sentiment_ts['policy_sentiment'] == 1
        dovish_days = sentiment_ts['policy_sentiment'] == -1
        neutral_days = sentiment_ts['policy_sentiment'] == 0
        
        # Calculate average volatility by sentiment
        hawkish_vol = vol_data[hawkish_days].mean() if hawkish_days.sum() > 0 else np.nan
        dovish_vol = vol_data[dovish_days].mean() if dovish_days.sum() > 0 else np.nan
        neutral_vol = vol_data[neutral_days].mean() if neutral_days.sum() > 0 else np.nan
        
        # Statistical tests
        if hawkish_days.sum() > 0 and neutral_days.sum() > 0:
            hawkish_vs_neutral = stats.ttest_ind(vol_data[hawkish_days], vol_data[neutral_days])
        else:
            hawkish_vs_neutral = (np.nan, np.nan)
        
        if dovish_days.sum() > 0 and neutral_days.sum() > 0:
            dovish_vs_neutral = stats.ttest_ind(vol_data[dovish_days], vol_data[neutral_days])
        else:
            dovish_vs_neutral = (np.nan, np.nan)
        
        results[stablecoin] = {
            'hawkish_volatility': hawkish_vol,
            'dovish_volatility': dovish_vol,
            'neutral_volatility': neutral_vol,
            'hawkish_vs_neutral_tstat': hawkish_vs_neutral[0],
            'hawkish_vs_neutral_pvalue': hawkish_vs_neutral[1],
            'dovish_vs_neutral_tstat': dovish_vs_neutral[0],
            'dovish_vs_neutral_pvalue': dovish_vs_neutral[1]
        }
        
        print(f"   {stablecoin}:")
        print(f"     Hawkish vol: {hawkish_vol:.4f}, Neutral vol: {neutral_vol:.4f}")
        print(f"     Dovish vol: {dovish_vol:.4f}")
        if not np.isnan(hawkish_vs_neutral[1]):
            print(f"     Hawkish vs Neutral p-value: {hawkish_vs_neutral[1]:.3f}")
    
    return results

def create_sentiment_volatility_visualization(volatility_df: pd.DataFrame, sentiment_ts: pd.DataFrame, correlations: Dict):
    """Create visualizations of sentiment-volatility relationships."""
    
    print("\n📈 Creating sentiment-volatility visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Policy Sentiment vs Stablecoin Volatility Analysis', fontsize=16, fontweight='bold')
    
    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    corr_data = []
    stablecoins = []
    for coin, corr_data_dict in correlations.items():
        if coin not in ['timestamp']:
            stablecoins.append(coin)
            corr_data.append([
                corr_data_dict['overall_correlation'],
                corr_data_dict['policy_event_correlation'],
                corr_data_dict['rolling_correlation_mean']
            ])
    
    corr_df = pd.DataFrame(corr_data, 
                          index=stablecoins,
                          columns=['Overall', 'Policy Events', 'Rolling Mean'])
    
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title('Sentiment-Volatility Correlations')
    
    # 2. Volatility by sentiment type
    ax2 = axes[0, 1]
    sentiment_labels = ['Hawkish', 'Neutral', 'Dovish']
    sentiment_values = [1, 0, -1]
    
    vol_by_sentiment = []
    for sentiment_val in sentiment_values:
        sentiment_mask = sentiment_ts['policy_sentiment'] == sentiment_val
        if sentiment_mask.sum() > 0:
            avg_vol = volatility_df.loc[sentiment_mask].mean().mean()
            vol_by_sentiment.append(avg_vol)
        else:
            vol_by_sentiment.append(np.nan)
    
    bars = ax2.bar(sentiment_labels, vol_by_sentiment, color=['red', 'gray', 'blue'], alpha=0.7)
    ax2.set_title('Average Volatility by Policy Sentiment')
    ax2.set_ylabel('Average Volatility')
    
    # Add value labels on bars
    for bar, value in zip(bars, vol_by_sentiment):
        if not np.isnan(value):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Time series of sentiment and volatility
    ax3 = axes[1, 0]
    
    # Plot average volatility
    avg_volatility = volatility_df.mean(axis=1)
    ax3.plot(avg_volatility.index, avg_volatility.values, label='Average Volatility', alpha=0.7)
    
    # Plot sentiment events
    policy_events = sentiment_ts[sentiment_ts['policy_sentiment'] != 0]
    for date, event in policy_events.iterrows():
        color = 'red' if event['policy_sentiment'] > 0 else 'blue'
        ax3.axvline(x=date, color=color, alpha=0.5, linestyle='--')
    
    ax3.set_title('Volatility and Policy Events Over Time')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    
    # 4. Scatter plot: sentiment vs volatility
    ax4 = axes[1, 1]
    
    # Use USDT as example
    if 'USDT' in volatility_df.columns:
        usdt_vol = volatility_df['USDT']
        sentiment_data = sentiment_ts['policy_sentiment']
        
        # Create scatter plot
        scatter = ax4.scatter(sentiment_data, usdt_vol, alpha=0.6, c=sentiment_data, cmap='RdBu_r')
        
        # Add trend line
        valid_data = ~(np.isnan(sentiment_data) | np.isnan(usdt_vol))
        if valid_data.sum() > 10:
            z = np.polyfit(sentiment_data[valid_data], usdt_vol[valid_data], 1)
            p = np.poly1d(z)
            ax4.plot(sentiment_data[valid_data], p(sentiment_data[valid_data]), "r--", alpha=0.8)
        
        ax4.set_xlabel('Policy Sentiment (-1=Dovish, 0=Neutral, 1=Hawkish)')
        ax4.set_ylabel('USDT Volatility')
        ax4.set_title('USDT Volatility vs Policy Sentiment')
    
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_volatility_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to data/processed/sentiment_volatility_analysis.png")

def explain_llm_methodology():
    """Explain how LLM interpretation works for quantitative research."""
    
    print("\n🧠 LLM Interpretation Methodology for Quantitative Research")
    print("=" * 70)
    
    print("""
📊 QUANTITATIVE RESEARCH FRAMEWORK:

1. TEXT PREPROCESSING:
   - Clean policy documents (remove HTML, normalize text)
   - Tokenize and truncate to model limits (4000 chars)
   - Extract key sections (policy statements, forward guidance)

2. SENTIMENT CLASSIFICATION:
   - Use DeepSeek to classify text as Hawkish/Dovish/Neutral
   - Generate confidence scores (0-1 scale)
   - Extract key phrases supporting classification
   - Provide explanations for transparency

3. QUANTITATIVE MAPPING:
   - Convert sentiment to numerical scores: Hawkish=1, Neutral=0, Dovish=-1
   - Weight by confidence: sentiment_score * confidence
   - Create time series aligned with market data

4. STATISTICAL ANALYSIS:
   - Correlation analysis: sentiment vs volatility/returns
   - Event studies: measure market response around policy announcements
   - Regression analysis: sentiment as predictor of market outcomes
   - Hypothesis testing: statistical significance of sentiment effects

5. VALIDATION & ROBUSTNESS:
   - Cross-validation with different LLM models
   - Manual verification of sentiment classifications
   - Sensitivity analysis with different confidence thresholds
   - Comparison with traditional sentiment indicators

🔬 RESEARCH METHODOLOGY:

A. EVENT STUDY APPROACH:
   - Identify policy announcement dates
   - Measure abnormal returns/volatility around events
   - Test statistical significance of market responses
   - Control for other market factors

B. TIME SERIES ANALYSIS:
   - Granger causality tests: does sentiment predict volatility?
   - VAR models: dynamic relationships between sentiment and markets
   - GARCH models: how sentiment affects volatility clustering

C. CROSS-SECTIONAL ANALYSIS:
   - Compare stablecoin responses to same policy events
   - Identify which stablecoins are most sensitive to policy
   - Analyze differences between centralized vs decentralized stablecoins

📈 QUANTITATIVE METRICS:

1. CORRELATION COEFFICIENTS:
   - Pearson correlation: linear relationship
   - Spearman correlation: monotonic relationship
   - Rolling correlations: time-varying relationships

2. REGRESSION ANALYSIS:
   - Volatility = α + β₁(Sentiment) + β₂(Market_Factors) + ε
   - Test significance of sentiment coefficient (β₁)
   - Control for market conditions, liquidity, etc.

3. EVENT STUDY METRICS:
   - Cumulative Abnormal Returns (CAR)
   - Buy-and-Hold Abnormal Returns (BHAR)
   - Abnormal volatility around events

4. STATISTICAL TESTS:
   - t-tests: compare volatility across sentiment categories
   - F-tests: test joint significance of sentiment variables
   - Bootstrap tests: non-parametric significance testing

🎯 RESEARCH QUESTIONS ADDRESSED:

1. Does policy sentiment predict stablecoin volatility?
2. Which stablecoins are most sensitive to policy changes?
3. How quickly do markets incorporate policy information?
4. Are there systematic differences in policy transmission?
5. Can sentiment analysis improve volatility forecasting?

💡 ADVANTAGES OF LLM-BASED APPROACH:

- Handles complex, nuanced policy language
- Captures context and subtle policy signals
- Scales to large volumes of policy documents
- Provides interpretable explanations
- Can be validated against expert classifications

⚠️ LIMITATIONS & CONSIDERATIONS:

- LLM bias and training data limitations
- Need for validation against expert judgment
- Computational costs for large-scale analysis
- Potential for overfitting to specific policy language
- Requires careful prompt engineering and validation
""")

def main():
    """Main analysis function."""
    
    print("🔬 Policy Sentiment vs Stablecoin Volatility Analysis")
    print("=" * 70)
    
    # Load data
    prices_df, volatility_df, returns_df = load_market_data()
    if volatility_df is None:
        return
    
    sentiment_df = load_policy_sentiment()
    if sentiment_df is None:
        return
    
    # Create sentiment time series
    sentiment_ts = create_sentiment_timeseries(sentiment_df, volatility_df.index)
    
    # Calculate correlations
    correlations = calculate_sentiment_volatility_correlation(volatility_df, sentiment_ts)
    
    # Analyze sentiment impact
    impact_results = analyze_sentiment_impact_on_volatility(volatility_df, sentiment_ts)
    
    # Create visualizations
    create_sentiment_volatility_visualization(volatility_df, sentiment_ts, correlations)
    
    # Explain methodology
    explain_llm_methodology()
    
    # Summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 30)
    print("✅ Sentiment-volatility correlations calculated")
    print("✅ Statistical significance tests performed")
    print("✅ Visualizations created")
    print("✅ Methodology documented")
    
    print("\n💡 KEY FINDINGS:")
    avg_correlation = np.mean([corr['overall_correlation'] for corr in correlations.values() if not np.isnan(corr['overall_correlation'])])
    print(f"   Average sentiment-volatility correlation: {avg_correlation:.3f}")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Add more policy events for robust analysis")
    print("   2. Implement event study methodology")
    print("   3. Build volatility forecasting models")
    print("   4. Analyze cross-stablecoin policy transmission")

if __name__ == "__main__":
    main()
