#!/usr/bin/env python3
"""
Simple data visualization tool for stablecoin analysis.
Creates basic plots to understand the data patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_price_evolution():
    """Plot price evolution over time."""
    
    print("📈 Creating price evolution plots...")
    
    try:
        df = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Stablecoin Price Evolution (2023-2025)', fontsize=16, fontweight='bold')
        
        coins = df.columns
        colors = plt.cm.Set3(np.linspace(0, 1, len(coins)))
        
        for i, coin in enumerate(coins):
            row = i // 4
            col = i % 4
            
            ax = axes[row, col]
            ax.plot(df.index, df[coin], color=colors[i], linewidth=2, label=coin)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Peg')
            ax.set_title(f'{coin} Price', fontweight='bold')
            ax.set_ylabel('Price (USD)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format y-axis to show small deviations
            ax.set_ylim(0.99, 1.01)
            
        plt.tight_layout()
        plt.savefig('data/processed/price_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Price evolution plot saved to data/processed/price_evolution.png")
        
    except Exception as e:
        print(f"❌ Error creating price plots: {e}")

def plot_volatility_comparison():
    """Plot volatility comparison across stablecoins."""
    
    print("📊 Creating volatility comparison...")
    
    try:
        df = pd.read_parquet('data/processed/volatility.parquet')
        
        # Calculate average volatility for each stablecoin
        avg_volatility = df.mean()
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(avg_volatility.index, avg_volatility.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(avg_volatility))))
        
        plt.title('Average Volatility by Stablecoin', fontsize=16, fontweight='bold')
        plt.xlabel('Stablecoin')
        plt.ylabel('Average Volatility')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_volatility.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data/processed/volatility_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Volatility comparison saved to data/processed/volatility_comparison.png")
        
    except Exception as e:
        print(f"❌ Error creating volatility plot: {e}")

def plot_peg_deviation_heatmap():
    """Create heatmap of peg deviations."""
    
    print("🎯 Creating peg deviation heatmap...")
    
    try:
        df = pd.read_parquet('data/processed/peg_deviations.parquet')
        
        # Sample data for heatmap (every 10th day to avoid overcrowding)
        sample_df = df.iloc[::10]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sample_df.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Peg Deviation'}, 
                   xticklabels=False)
        
        plt.title('Peg Deviation Heatmap (Sample)', fontsize=16, fontweight='bold')
        plt.xlabel('Time (Sample Points)')
        plt.ylabel('Stablecoin')
        
        plt.tight_layout()
        plt.savefig('data/processed/peg_deviation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Peg deviation heatmap saved to data/processed/peg_deviation_heatmap.png")
        
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")

def plot_correlation_matrix():
    """Plot correlation matrix of stablecoin returns."""
    
    print("🔗 Creating correlation matrix...")
    
    try:
        df = pd.read_parquet('data/processed/returns.parquet')
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        
        plt.title('Stablecoin Returns Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Correlation matrix saved to data/processed/correlation_matrix.png")
        
    except Exception as e:
        print(f"❌ Error creating correlation plot: {e}")

def create_summary_dashboard():
    """Create a summary dashboard with key metrics."""
    
    print("📊 Creating summary dashboard...")
    
    try:
        # Load data
        prices = pd.read_parquet('data/processed/stablecoin_prices.parquet')
        volatility = pd.read_parquet('data/processed/volatility.parquet')
        peg_deviations = pd.read_parquet('data/processed/peg_deviations.parquet')
        
        # Create summary statistics
        summary_stats = pd.DataFrame({
            'Current_Price': prices.iloc[-1],
            'Avg_Price': prices.mean(),
            'Price_Std': prices.std(),
            'Max_Peg_Deviation': peg_deviations.abs().max(),
            'Avg_Volatility': volatility.mean(),
            'Max_Volatility': volatility.max()
        })
        
        # Create dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stablecoin Analysis Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Current prices vs perfect peg
        ax1 = axes[0, 0]
        current_prices = prices.iloc[-1]
        deviations = current_prices - 1.0
        colors = ['red' if d > 0 else 'blue' for d in deviations]
        bars = ax1.bar(deviations.index, deviations.values, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax1.set_title('Current Peg Deviations', fontweight='bold')
        ax1.set_ylabel('Deviation from $1.00')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, deviations.values):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.0001 if value > 0 else -0.0001),
                    f'{value:.4f}', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold')
        
        # 2. Volatility comparison
        ax2 = axes[0, 1]
        avg_vol = volatility.mean()
        ax2.bar(avg_vol.index, avg_vol.values, color=plt.cm.viridis(np.linspace(0, 1, len(avg_vol))))
        ax2.set_title('Average Volatility', fontweight='bold')
        ax2.set_ylabel('Volatility')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Price stability ranking
        ax3 = axes[1, 0]
        stability_score = 1 / (peg_deviations.abs().mean() + 0.0001)  # Higher = more stable
        stability_ranked = stability_score.sort_values(ascending=True)
        ax3.barh(stability_ranked.index, stability_ranked.values, 
                color=plt.cm.Greens(np.linspace(0.3, 1, len(stability_ranked))))
        ax3.set_title('Stability Ranking (Higher = More Stable)', fontweight='bold')
        ax3.set_xlabel('Stability Score')
        
        # 4. Risk assessment
        ax4 = axes[1, 1]
        risk_score = volatility.mean() * peg_deviations.abs().mean() * 1000  # Combined risk metric
        risk_ranked = risk_score.sort_values(ascending=False)
        ax4.bar(risk_ranked.index, risk_ranked.values, 
               color=plt.cm.Reds(np.linspace(0.3, 1, len(risk_ranked))))
        ax4.set_title('Risk Assessment (Higher = Riskier)', fontweight='bold')
        ax4.set_ylabel('Risk Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/processed/summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Summary dashboard saved to data/processed/summary_dashboard.png")
        
        # Print summary table
        print("\n📊 Summary Statistics:")
        print("=" * 80)
        print(summary_stats.round(6))
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")

def main():
    """Main visualization function."""
    
    print("🎨 Stablecoin Data Visualization Tool")
    print("=" * 50)
    
    # Check if data exists
    if not Path('data/processed/stablecoin_prices.parquet').exists():
        print("❌ No processed data found. Run the pipeline first:")
        print("   run_all_windows.bat")
        return
    
    # Create visualizations
    plot_price_evolution()
    plot_volatility_comparison()
    plot_peg_deviation_heatmap()
    plot_correlation_matrix()
    create_summary_dashboard()
    
    print("\n🎉 Visualization complete!")
    print("\n📁 Generated plots:")
    print("   - price_evolution.png")
    print("   - volatility_comparison.png") 
    print("   - peg_deviation_heatmap.png")
    print("   - correlation_matrix.png")
    print("   - summary_dashboard.png")
    print("\n💡 All plots saved to data/processed/")

if __name__ == "__main__":
    main()
