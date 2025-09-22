#!/usr/bin/env python3
"""
Analyze correlation between policy sentiment and stablecoin peg deviations.
Focus on how policy sentiment affects peg stability rather than volatility.
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

def load_market_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load stablecoin market data."""
    try:
        prices_df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        peg_deviations_df = pd.read_parquet('data/processed/peg_deviations.parquet')
        returns_df = pd.read_parquet('data/processed/returns.parquet')
        
        print(f"✅ Loaded market data:")
        print(f"   Prices: {prices_df.shape}")
        print(f"   Peg Deviations: {peg_deviations_df.shape}")
        print(f"   Returns: {returns_df.shape}")
        
        return prices_df, peg_deviations_df, returns_df
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

def calculate_sentiment_peg_correlation(peg_deviations_df: pd.DataFrame, sentiment_ts: pd.DataFrame) -> Dict:
    """Calculate correlation between sentiment and peg deviations."""
    
    print("\n📊 Calculating sentiment-peg deviation correlations...")
    
    correlations = {}
    
    for stablecoin in peg_deviations_df.columns:
        if stablecoin in ['timestamp']:
            continue
            
        # Align peg deviations and sentiment data
        peg_data = peg_deviations_df[stablecoin]
        sent_data = sentiment_ts['policy_sentiment']
        
        # Calculate correlations
        correlation = peg_data.corr(sent_data)
        
        # Calculate rolling correlation (30-day window)
        rolling_corr = peg_data.rolling(window=30).corr(sent_data)
        
        # Calculate correlation during policy events only
        policy_days = sentiment_ts['policy_sentiment'] != 0
        if policy_days.sum() > 0:
            policy_correlation = peg_data[policy_days].corr(sent_data[policy_days])
        else:
            policy_correlation = np.nan
        
        # Calculate absolute peg deviation correlation (magnitude of deviation)
        abs_peg_data = abs(peg_data)
        abs_correlation = abs_peg_data.corr(sent_data)
        
        correlations[stablecoin] = {
            'overall_correlation': correlation,
            'policy_event_correlation': policy_correlation,
            'abs_deviation_correlation': abs_correlation,
            'rolling_correlation_mean': rolling_corr.mean(),
            'rolling_correlation_std': rolling_corr.std()
        }
        
        print(f"   {stablecoin}: Overall={correlation:.3f}, Abs Dev={abs_correlation:.3f}, Policy Events={policy_correlation:.3f}")
    
    return correlations

def analyze_sentiment_impact_on_peg(peg_deviations_df: pd.DataFrame, sentiment_ts: pd.DataFrame) -> Dict:
    """Analyze how sentiment impacts peg stability."""
    
    print("\n🎯 Analyzing sentiment impact on peg stability...")
    
    results = {}
    
    for stablecoin in peg_deviations_df.columns:
        if stablecoin in ['timestamp']:
            continue
            
        peg_data = peg_deviations_df[stablecoin]
        abs_peg_data = abs(peg_data)
        
        # Separate peg deviations by sentiment
        hawkish_days = sentiment_ts['policy_sentiment'] == 1
        dovish_days = sentiment_ts['policy_sentiment'] == -1
        neutral_days = sentiment_ts['policy_sentiment'] == 0
        
        # Calculate average peg deviations by sentiment
        hawkish_peg = peg_data[hawkish_days].mean() if hawkish_days.sum() > 0 else np.nan
        dovish_peg = peg_data[dovish_days].mean() if dovish_days.sum() > 0 else np.nan
        neutral_peg = peg_data[neutral_days].mean() if neutral_days.sum() > 0 else np.nan
        
        # Calculate average absolute peg deviations (stability measure)
        hawkish_abs_peg = abs_peg_data[hawkish_days].mean() if hawkish_days.sum() > 0 else np.nan
        dovish_abs_peg = abs_peg_data[dovish_days].mean() if dovish_days.sum() > 0 else np.nan
        neutral_abs_peg = abs_peg_data[neutral_days].mean() if neutral_days.sum() > 0 else np.nan
        
        # Statistical tests for absolute deviations (stability)
        if hawkish_days.sum() > 0 and neutral_days.sum() > 0:
            hawkish_vs_neutral = stats.ttest_ind(abs_peg_data[hawkish_days], abs_peg_data[neutral_days])
        else:
            hawkish_vs_neutral = (np.nan, np.nan)
        
        if dovish_days.sum() > 0 and neutral_days.sum() > 0:
            dovish_vs_neutral = stats.ttest_ind(abs_peg_data[dovish_days], abs_peg_data[neutral_days])
        else:
            dovish_vs_neutral = (np.nan, np.nan)
        
        # Calculate peg stability metrics
        peg_stability_hawkish = 1 - hawkish_abs_peg if not np.isnan(hawkish_abs_peg) else np.nan
        peg_stability_dovish = 1 - dovish_abs_peg if not np.isnan(dovish_abs_peg) else np.nan
        peg_stability_neutral = 1 - neutral_abs_peg if not np.isnan(neutral_abs_peg) else np.nan
        
        results[stablecoin] = {
            'hawkish_peg_deviation': hawkish_peg,
            'dovish_peg_deviation': dovish_peg,
            'neutral_peg_deviation': neutral_peg,
            'hawkish_abs_deviation': hawkish_abs_peg,
            'dovish_abs_deviation': dovish_abs_peg,
            'neutral_abs_deviation': neutral_abs_peg,
            'hawkish_peg_stability': peg_stability_hawkish,
            'dovish_peg_stability': peg_stability_dovish,
            'neutral_peg_stability': peg_stability_neutral,
            'hawkish_vs_neutral_tstat': hawkish_vs_neutral[0],
            'hawkish_vs_neutral_pvalue': hawkish_vs_neutral[1],
            'dovish_vs_neutral_tstat': dovish_vs_neutral[0],
            'dovish_vs_neutral_pvalue': dovish_vs_neutral[1]
        }
        
        print(f"   {stablecoin}:")
        print(f"     Hawkish: {hawkish_peg:.6f} (abs: {hawkish_abs_peg:.6f}, stability: {peg_stability_hawkish:.6f})")
        print(f"     Neutral: {neutral_peg:.6f} (abs: {neutral_abs_peg:.6f}, stability: {peg_stability_neutral:.6f})")
        print(f"     Dovish: {dovish_peg:.6f} (abs: {dovish_abs_peg:.6f}, stability: {peg_stability_dovish:.6f})")
        if not np.isnan(hawkish_vs_neutral[1]):
            print(f"     Hawkish vs Neutral p-value: {hawkish_vs_neutral[1]:.3f}")
    
    return results

def analyze_peg_stress_events(peg_deviations_df: pd.DataFrame, sentiment_ts: pd.DataFrame) -> Dict:
    """Analyze peg stress events around policy announcements."""
    
    print("\n🚨 Analyzing peg stress events around policy announcements...")
    
    # Define peg stress threshold (1% deviation from $1.00)
    stress_threshold = 0.01
    
    stress_events = {}
    
    for stablecoin in peg_deviations_df.columns:
        if stablecoin in ['timestamp']:
            continue
            
        peg_data = peg_deviations_df[stablecoin]
        abs_peg_data = abs(peg_data)
        
        # Find peg stress events
        stress_mask = abs_peg_data > stress_threshold
        stress_dates = peg_data[stress_mask].index
        
        # Check if stress events occur around policy announcements
        policy_dates = sentiment_ts[sentiment_ts['policy_sentiment'] != 0].index
        
        stress_around_policy = 0
        for policy_date in policy_dates:
            # Check ±3 days around policy announcement
            window_start = policy_date - pd.Timedelta(days=3)
            window_end = policy_date + pd.Timedelta(days=3)
            
            stress_in_window = stress_dates[(stress_dates >= window_start) & (stress_dates <= window_end)]
            if len(stress_in_window) > 0:
                stress_around_policy += len(stress_in_window)
        
        stress_events[stablecoin] = {
            'total_stress_events': len(stress_dates),
            'stress_around_policy': stress_around_policy,
            'stress_rate_around_policy': stress_around_policy / len(policy_dates) if len(policy_dates) > 0 else 0,
            'max_deviation': abs_peg_data.max(),
            'avg_stress_deviation': abs_peg_data[stress_mask].mean() if stress_mask.sum() > 0 else 0
        }
        
        print(f"   {stablecoin}: {len(stress_dates)} stress events, {stress_around_policy} around policy")
    
    return stress_events

def create_peg_visualization(peg_deviations_df: pd.DataFrame, sentiment_ts: pd.DataFrame, correlations: Dict):
    """Create visualizations of sentiment-peg relationships."""
    
    print("\n📈 Creating sentiment-peg deviation visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Policy Sentiment vs Stablecoin Peg Stability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    corr_data = []
    stablecoins = []
    for coin, corr_data_dict in correlations.items():
        if coin not in ['timestamp']:
            stablecoins.append(coin)
            corr_data.append([
                corr_data_dict['overall_correlation'],
                corr_data_dict['abs_deviation_correlation'],
                corr_data_dict['rolling_correlation_mean']
            ])
    
    corr_df = pd.DataFrame(corr_data, 
                          index=stablecoins,
                          columns=['Overall', 'Abs Deviation', 'Rolling Mean'])
    
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title('Sentiment-Peg Deviation Correlations')
    
    # 2. Peg stability by sentiment type
    ax2 = axes[0, 1]
    
    # Calculate average absolute peg deviations by sentiment
    sentiment_labels = ['Hawkish', 'Neutral', 'Dovish']
    sentiment_values = [1, 0, -1]
    
    peg_stability_by_sentiment = []
    for sentiment_val in sentiment_values:
        sentiment_mask = sentiment_ts['policy_sentiment'] == sentiment_val
        if sentiment_mask.sum() > 0:
            avg_abs_dev = abs(peg_deviations_df.loc[sentiment_mask]).mean().mean()
            peg_stability = 1 - avg_abs_dev  # Higher stability = lower deviation
            peg_stability_by_sentiment.append(peg_stability)
        else:
            peg_stability_by_sentiment.append(np.nan)
    
    bars = ax2.bar(sentiment_labels, peg_stability_by_sentiment, color=['red', 'gray', 'blue'], alpha=0.7)
    ax2.set_title('Peg Stability by Policy Sentiment')
    ax2.set_ylabel('Peg Stability (1 - Avg Abs Deviation)')
    ax2.set_ylim(0.99, 1.0)  # Focus on high stability range
    
    # Add value labels on bars
    for bar, value in zip(bars, peg_stability_by_sentiment):
        if not np.isnan(value):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Time series of peg deviations and sentiment events
    ax3 = axes[1, 0]
    
    # Plot average absolute peg deviation
    avg_abs_deviation = abs(peg_deviations_df).mean(axis=1)
    ax3.plot(avg_abs_deviation.index, avg_abs_deviation.values, label='Avg Abs Peg Deviation', alpha=0.7, color='blue')
    
    # Add peg stress threshold line
    ax3.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Stress Threshold (1%)')
    
    # Plot sentiment events
    policy_events = sentiment_ts[sentiment_ts['policy_sentiment'] != 0]
    for date, event in policy_events.iterrows():
        color = 'red' if event['policy_sentiment'] > 0 else 'blue'
        ax3.axvline(x=date, color=color, alpha=0.5, linestyle='--')
    
    ax3.set_title('Peg Deviations and Policy Events Over Time')
    ax3.set_ylabel('Absolute Peg Deviation')
    ax3.legend()
    
    # 4. Scatter plot: sentiment vs peg deviation (USDT example)
    ax4 = axes[1, 1]
    
    # Use USDT as example
    if 'USDT' in peg_deviations_df.columns:
        usdt_peg = peg_deviations_df['USDT']
        sentiment_data = sentiment_ts['policy_sentiment']
        
        # Create scatter plot
        scatter = ax4.scatter(sentiment_data, usdt_peg, alpha=0.6, c=sentiment_data, cmap='RdBu_r')
        
        # Add trend line
        valid_data = ~(np.isnan(sentiment_data) | np.isnan(usdt_peg))
        if valid_data.sum() > 10:
            z = np.polyfit(sentiment_data[valid_data], usdt_peg[valid_data], 1)
            p = np.poly1d(z)
            ax4.plot(sentiment_data[valid_data], p(sentiment_data[valid_data]), "r--", alpha=0.8)
        
        ax4.set_xlabel('Policy Sentiment (-1=Dovish, 0=Neutral, 1=Hawkish)')
        ax4.set_ylabel('USDT Peg Deviation from $1.00')
        ax4.set_title('USDT Peg Deviation vs Policy Sentiment')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_peg_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to data/processed/sentiment_peg_analysis.png")

def main():
    """Main analysis function."""
    
    print("🎯 Policy Sentiment vs Stablecoin Peg Stability Analysis")
    print("=" * 70)
    
    # Load data
    prices_df, peg_deviations_df, returns_df = load_market_data()
    if peg_deviations_df is None:
        return
    
    sentiment_df = load_policy_sentiment()
    if sentiment_df is None:
        return
    
    # Create sentiment time series
    sentiment_ts = create_sentiment_timeseries(sentiment_df, peg_deviations_df.index)
    
    # Calculate correlations
    correlations = calculate_sentiment_peg_correlation(peg_deviations_df, sentiment_ts)
    
    # Analyze sentiment impact on peg
    impact_results = analyze_sentiment_impact_on_peg(peg_deviations_df, sentiment_ts)
    
    # Analyze peg stress events
    stress_events = analyze_peg_stress_events(peg_deviations_df, sentiment_ts)
    
    # Create visualizations
    create_peg_visualization(peg_deviations_df, sentiment_ts, correlations)
    
    # Summary
    print("\n📊 PEG STABILITY ANALYSIS SUMMARY")
    print("=" * 50)
    print("✅ Sentiment-peg deviation correlations calculated")
    print("✅ Peg stability by sentiment analyzed")
    print("✅ Peg stress events around policy announcements identified")
    print("✅ Visualizations created")
    
    print("\n💡 KEY FINDINGS:")
    avg_correlation = np.mean([corr['overall_correlation'] for corr in correlations.values() if not np.isnan(corr['overall_correlation'])])
    avg_abs_correlation = np.mean([corr['abs_deviation_correlation'] for corr in correlations.values() if not np.isnan(corr['abs_deviation_correlation'])])
    print(f"   Average sentiment-peg correlation: {avg_correlation:.3f}")
    print(f"   Average sentiment-abs deviation correlation: {avg_abs_correlation:.3f}")
    
    print("\n🎯 PEG STABILITY INSIGHTS:")
    print("   - Lower absolute deviation = higher peg stability")
    print("   - Peg stress events (>1% deviation) around policy announcements")
    print("   - Sentiment impact on peg maintenance mechanisms")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Add more policy events for robust peg analysis")
    print("   2. Analyze peg recovery patterns after policy events")
    print("   3. Study arbitrage mechanisms during peg stress")
    print("   4. Compare centralized vs decentralized peg stability")

if __name__ == "__main__":
    main()
