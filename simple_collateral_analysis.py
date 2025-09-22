#!/usr/bin/env python3
"""
Simplified analysis of crypto collateral dynamics for stablecoins.
Focus on key insights about how crypto assets affect stablecoin stability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Dict, List
import sys
from pathlib import Path

def fetch_crypto_data():
    """Fetch data for key crypto assets used as stablecoin collateral."""
    
    print("🪙 Fetching Crypto Collateral Data")
    print("=" * 50)
    
    # Key crypto assets used as collateral
    crypto_assets = {
        'ETH': 'Ethereum (primary collateral for DAI, LUSD)',
        'BTC': 'Bitcoin (collateral for some DAI mechanisms)',
        'BNB': 'Binance Coin (backing for BUSD)',
        'SOL': 'Solana (used in some stablecoin mechanisms)',
        'AVAX': 'Avalanche (collateral in some protocols)',
        'MATIC': 'Polygon (used in DeFi protocols)'
    }
    
    crypto_data = {}
    
    for symbol, description in crypto_assets.items():
        try:
            print(f"📊 Fetching {symbol} data...")
            
            ticker = yf.Ticker(f"{symbol}-USD")
            hist = ticker.history(period="2y", interval="1d")
            
            if not hist.empty:
                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['description'] = description
                
                crypto_data[symbol] = hist
                print(f"   ✅ {symbol}: {len(hist)} days of data")
            else:
                print(f"   ❌ {symbol}: No data available")
                
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    return crypto_data

def analyze_collateral_mechanisms():
    """Analyze different stablecoin collateral mechanisms."""
    
    print("\n⚙️ Stablecoin Collateral Mechanisms")
    print("=" * 50)
    
    mechanisms = {
        'DAI': {
            'type': 'Over-collateralized',
            'collateral_ratio': '150%+',
            'primary_collateral': 'ETH',
            'mechanism': 'MakerDAO CDP',
            'stability_features': ['Liquidation at 150%', 'Stability fees', 'Emergency shutdown'],
            'risk_factors': ['ETH volatility', 'Liquidation cascades', 'Governance risk']
        },
        'LUSD': {
            'type': 'Over-collateralized',
            'collateral_ratio': '110%+',
            'primary_collateral': 'ETH',
            'mechanism': 'Liquity Protocol',
            'stability_features': ['Liquidation at 110%', 'No governance token', 'Decentralized'],
            'risk_factors': ['ETH volatility', 'Liquidation risk', 'Protocol risk']
        },
        'FRAX': {
            'type': 'Fractional',
            'collateral_ratio': 'Variable (80-100%)',
            'primary_collateral': 'USDC + ETH + FXS',
            'mechanism': 'Algorithmic + Collateral hybrid',
            'stability_features': ['Dynamic collateral ratio', 'FXS governance', 'AMO'],
            'risk_factors': ['USDC depeg risk', 'ETH volatility', 'Algorithmic risk']
        },
        'USDC': {
            'type': 'Fiat-backed',
            'collateral_ratio': '100%',
            'primary_collateral': 'USD',
            'mechanism': 'Centralized reserves',
            'stability_features': ['Bank deposits', 'Treasury bills', 'Regular audits'],
            'risk_factors': ['Counterparty risk', 'Regulatory risk', 'Banking risk']
        },
        'USDT': {
            'type': 'Fiat-backed',
            'collateral_ratio': '100%',
            'primary_collateral': 'USD',
            'mechanism': 'Centralized reserves',
            'stability_features': ['Bank deposits', 'Commercial paper', 'Limited transparency'],
            'risk_factors': ['Counterparty risk', 'Transparency risk', 'Regulatory risk']
        }
    }
    
    for stablecoin, mechanism in mechanisms.items():
        print(f"\n🪙 {stablecoin} Mechanism:")
        print(f"   Type: {mechanism['type']}")
        print(f"   Collateral Ratio: {mechanism['collateral_ratio']}")
        print(f"   Primary Collateral: {mechanism['primary_collateral']}")
        print(f"   Mechanism: {mechanism['mechanism']}")
        print(f"   Stability Features: {', '.join(mechanism['stability_features'])}")
        print(f"   Risk Factors: {', '.join(mechanism['risk_factors'])}")
    
    return mechanisms

def analyze_crypto_volatility(crypto_data: Dict[str, pd.DataFrame]):
    """Analyze volatility patterns in crypto collateral assets."""
    
    print("\n📊 Crypto Collateral Volatility Analysis")
    print("=" * 50)
    
    volatility_analysis = {}
    
    for symbol, df in crypto_data.items():
        print(f"\n📈 {symbol} Volatility Analysis:")
        
        if len(df) == 0:
            continue
            
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        returns = df['returns'].dropna()
        
        # Calculate volatility metrics
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(365)
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Value at Risk
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Calculate tail risk
        tail_risk_95 = returns[returns <= var_95].mean()
        tail_risk_99 = returns[returns <= var_99].mean()
        
        volatility_analysis[symbol] = {
            'daily_volatility': daily_volatility,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'tail_risk_95': tail_risk_95,
            'tail_risk_99': tail_risk_99
        }
        
        print(f"   Daily Volatility: {daily_volatility:.4f}")
        print(f"   Annualized Volatility: {annualized_volatility:.2%}")
        print(f"   Maximum Drawdown: {max_drawdown:.2%}")
        print(f"   VaR (95%): {var_95:.2%}")
        print(f"   VaR (99%): {var_99:.2%}")
        print(f"   Tail Risk (95%): {tail_risk_95:.2%}")
        print(f"   Tail Risk (99%): {tail_risk_99:.2%}")
    
    return volatility_analysis

def analyze_collateral_correlations(crypto_data: Dict[str, pd.DataFrame]):
    """Analyze correlations between different crypto collateral assets."""
    
    print("\n🔗 Crypto Collateral Correlations")
    print("=" * 40)
    
    # Prepare price data for correlation analysis
    price_data = {}
    
    for symbol, df in crypto_data.items():
        if len(df) > 0:
            price_data[symbol] = df.set_index('Date')['Close']
    
    if len(price_data) < 2:
        print("❌ Insufficient data for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_df = pd.DataFrame(price_data).corr()
    
    print("📊 Price Correlations:")
    print(corr_df.round(3))
    
    # Analyze return correlations
    return_data = {}
    for symbol, df in crypto_data.items():
        if len(df) > 0:
            df['returns'] = df['Close'].pct_change()
            return_data[symbol] = df.set_index('Date')['returns'].dropna()
    
    if len(return_data) >= 2:
        return_corr_df = pd.DataFrame(return_data).corr()
        print("\n📈 Return Correlations:")
        print(return_corr_df.round(3))
    
    return corr_df

def create_collateral_visualization(crypto_data: Dict[str, pd.DataFrame], volatility_analysis: Dict):
    """Create visualizations of crypto collateral dynamics."""
    
    print("\n📈 Creating Collateral Dynamics Visualizations")
    print("=" * 50)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Crypto Collateral Dynamics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price evolution
    ax1 = axes[0, 0]
    for symbol, df in crypto_data.items():
        if len(df) > 0:
            ax1.plot(df['Date'], df['Close'], label=symbol, alpha=0.7, linewidth=2)
    ax1.set_title('Crypto Collateral Asset Price Evolution')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility comparison
    ax2 = axes[0, 1]
    symbols = list(volatility_analysis.keys())
    volatilities = [volatility_analysis[symbol]['annualized_volatility'] for symbol in symbols]
    
    bars = ax2.bar(symbols, volatilities, color='skyblue', alpha=0.7)
    ax2.set_title('Annualized Volatility Comparison')
    ax2.set_ylabel('Annualized Volatility')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, volatilities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Maximum drawdown comparison
    ax3 = axes[1, 0]
    drawdowns = [volatility_analysis[symbol]['max_drawdown'] for symbol in symbols]
    
    bars = ax3.bar(symbols, drawdowns, color='lightcoral', alpha=0.7)
    ax3.set_title('Maximum Drawdown Comparison')
    ax3.set_ylabel('Maximum Drawdown')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, drawdowns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Risk-return scatter
    ax4 = axes[1, 1]
    
    # Calculate average returns
    returns = []
    for symbol in symbols:
        df = crypto_data[symbol]
        if len(df) > 0:
            avg_return = df['Close'].pct_change().mean() * 365  # Annualized
            returns.append(avg_return)
        else:
            returns.append(0)
    
    scatter = ax4.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(symbols)), cmap='viridis')
    
    # Add labels
    for i, symbol in enumerate(symbols):
        ax4.annotate(symbol, (volatilities[i], returns[i]), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    
    ax4.set_title('Risk-Return Profile')
    ax4.set_xlabel('Annualized Volatility')
    ax4.set_ylabel('Annualized Return')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/crypto_collateral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to data/processed/crypto_collateral_analysis.png")

def analyze_stablecoin_collateral_relationships():
    """Analyze how different stablecoins relate to their collateral assets."""
    
    print("\n🔍 Stablecoin-Collateral Relationships")
    print("=" * 50)
    
    relationships = {
        'DAI': {
            'collateral_dependency': 'High',
            'primary_risk': 'ETH volatility',
            'stability_mechanism': 'Over-collateralization (150%+)',
            'liquidation_risk': 'Medium-High',
            'governance_risk': 'Medium',
            'decentralization': 'High'
        },
        'LUSD': {
            'collateral_dependency': 'Very High',
            'primary_risk': 'ETH volatility',
            'stability_mechanism': 'Over-collateralization (110%+)',
            'liquidation_risk': 'High',
            'governance_risk': 'Low',
            'decentralization': 'Very High'
        },
        'FRAX': {
            'collateral_dependency': 'Medium',
            'primary_risk': 'USDC depeg + ETH volatility',
            'stability_mechanism': 'Fractional + Algorithmic',
            'liquidation_risk': 'Low-Medium',
            'governance_risk': 'Medium',
            'decentralization': 'Medium'
        },
        'USDC': {
            'collateral_dependency': 'None (fiat-backed)',
            'primary_risk': 'Counterparty risk',
            'stability_mechanism': 'Centralized reserves',
            'liquidation_risk': 'None',
            'governance_risk': 'High',
            'decentralization': 'None'
        },
        'USDT': {
            'collateral_dependency': 'None (fiat-backed)',
            'primary_risk': 'Counterparty + transparency risk',
            'stability_mechanism': 'Centralized reserves',
            'liquidation_risk': 'None',
            'governance_risk': 'High',
            'decentralization': 'None'
        }
    }
    
    for stablecoin, relationship in relationships.items():
        print(f"\n🪙 {stablecoin} Collateral Analysis:")
        print(f"   Collateral Dependency: {relationship['collateral_dependency']}")
        print(f"   Primary Risk: {relationship['primary_risk']}")
        print(f"   Stability Mechanism: {relationship['stability_mechanism']}")
        print(f"   Liquidation Risk: {relationship['liquidation_risk']}")
        print(f"   Governance Risk: {relationship['governance_risk']}")
        print(f"   Decentralization: {relationship['decentralization']}")
    
    return relationships

def main():
    """Main analysis function."""
    
    print("🪙 Crypto Collateral Dynamics Analysis")
    print("=" * 60)
    
    # Fetch crypto data
    crypto_data = fetch_crypto_data()
    
    if not crypto_data:
        print("❌ No crypto data available for analysis")
        return
    
    # Analyze mechanisms
    mechanisms = analyze_collateral_mechanisms()
    
    # Analyze volatility
    volatility_analysis = analyze_crypto_volatility(crypto_data)
    
    # Analyze correlations
    correlation_analysis = analyze_collateral_correlations(crypto_data)
    
    # Analyze relationships
    relationships = analyze_stablecoin_collateral_relationships()
    
    # Create visualizations
    create_collateral_visualization(crypto_data, volatility_analysis)
    
    # Summary
    print("\n📊 CRYPTO COLLATERAL ANALYSIS SUMMARY")
    print("=" * 50)
    print("✅ Crypto collateral data fetched and analyzed")
    print("✅ Stablecoin mechanisms examined")
    print("✅ Volatility patterns identified")
    print("✅ Correlation structures analyzed")
    print("✅ Risk relationships mapped")
    print("✅ Comprehensive visualizations created")
    
    print("\n💡 KEY INSIGHTS:")
    print("   - ETH is the primary collateral for decentralized stablecoins")
    print("   - Over-collateralization provides stability but requires capital efficiency")
    print("   - Crypto volatility directly impacts stablecoin stability mechanisms")
    print("   - Different stablecoin types have varying collateral dependencies")
    print("   - Liquidation risks are highest for over-collateralized stablecoins")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Analyze specific liquidation events and their impact")
    print("   2. Study collateral efficiency and capital utilization")
    print("   3. Examine cross-chain collateral dynamics")
    print("   4. Investigate policy impact on collateral assets")
    print("   5. Model collateral stress scenarios")

if __name__ == "__main__":
    main()
