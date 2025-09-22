# 📊 Stablecoin Data Interpretation Guide

## 🎯 **What This Data Tells Us**

Your stablecoin policy research project has generated **732 days** of comprehensive data (Sept 2023 - Sept 2025) across **8 major stablecoins**. Here's how to understand what you're looking at:

---

## 📈 **Key Findings from Your Data**

### 🏆 **Most Stable Stablecoins (Best Peg Maintenance)**
1. **USDC**: Avg deviation 0.000101 (99.99% accurate peg)
2. **DAI**: Avg deviation 0.000146 (99.99% accurate peg)  
3. **USDT**: Avg deviation 0.000385 (99.96% accurate peg)
4. **USDP**: Avg deviation 0.000927 (99.91% accurate peg)

### ⚠️ **Most Volatile Stablecoins (Highest Risk)**
1. **LUSD**: Avg volatility 0.062072 (6.2% daily volatility)
2. **BUSD**: Avg volatility 0.019279 (1.9% daily volatility)
3. **TUSD**: Avg volatility 0.024908 (2.5% daily volatility)

### 🚨 **Peg Stress Events Identified**
- **LUSD**: Major depeg to $0.938 (6.1% below peg)
- **BUSD**: Significant premium to $1.090 (9% above peg)
- **TUSD**: Depeg to $0.960 (4% below peg)

---

## 📊 **Data Files Explained**

### 💰 **Price Data (`stablecoin_prices.parquet`)**
- **What it shows**: Daily closing prices for each stablecoin
- **How to read**: Values close to 1.0 = good peg maintenance
- **Key insight**: USDC and DAI show the tightest peg adherence

### 📈 **Volatility Data (`volatility.parquet`)**
- **What it shows**: Rolling volatility (price instability) over time
- **How to read**: Lower values = more stable, higher = riskier
- **Key insight**: LUSD shows 6x higher volatility than USDC

### 🎯 **Peg Deviations (`peg_deviations.parquet`)**
- **What it shows**: How far each price deviates from $1.00
- **How to read**: 
  - Positive = trading above peg (premium)
  - Negative = trading below peg (discount)
- **Key insight**: FRAX trades below peg 93% of the time

### 🔗 **Correlations (`rolling_correlation.parquet`)**
- **What it shows**: How stablecoins move together
- **How to read**: 
  - High correlation (>0.7) = systemic risk
  - Low correlation (<0.3) = independent behavior
- **Key insight**: Most stablecoins show moderate correlation (0.4-0.6)

### 📊 **Market Depth (`market_depth.parquet`)**
- **What it shows**: Liquidity and trading depth measures
- **How to read**: Higher values = more liquid markets
- **Key insight**: USDT and USDC show highest market depth

---

## 🔍 **How to Interpret the Visualizations**

### 📈 **Price Evolution Plots**
- **Red dashed line**: Perfect $1.00 peg
- **Blue/colored lines**: Actual stablecoin prices
- **Tight bands**: Good peg maintenance
- **Wide swings**: Peg stress events

### 📊 **Volatility Comparison**
- **Lower bars**: More stable stablecoins
- **Higher bars**: Riskier stablecoins
- **USDC/DAI**: Consistently lowest volatility
- **LUSD**: Significantly higher volatility

### 🎯 **Peg Deviation Heatmap**
- **Red**: Trading above peg (premium)
- **Blue**: Trading below peg (discount)
- **White**: Close to perfect peg
- **Patterns**: Show systematic deviations over time

### 🔗 **Correlation Matrix**
- **Red**: Positive correlation (move together)
- **Blue**: Negative correlation (move opposite)
- **Values**: -1 to +1 scale
- **High values**: Systemic risk indicators

---

## 🎯 **Policy Research Insights**

### 📊 **Stability Rankings**
1. **USDC**: Most stable, lowest volatility
2. **DAI**: Second most stable, algorithmic backing
3. **USDT**: Third most stable, largest market cap
4. **USDP**: Fourth most stable, Paxos-backed
5. **FRAX**: Moderate stability, fractional algorithm
6. **TUSD**: Higher volatility, some peg stress
7. **BUSD**: High volatility, discontinued
8. **LUSD**: Highest volatility, most peg stress

### 🚨 **Risk Assessment**
- **Systemic Risk**: Moderate correlation suggests some systemic risk
- **Individual Risk**: LUSD and BUSD show highest individual risk
- **Market Stress**: Multiple depeg events identified
- **Liquidity**: USDT/USDC maintain highest market depth

### 📈 **Market Dynamics**
- **Algorithmic vs Collateralized**: DAI (algorithmic) performs well
- **Centralized vs Decentralized**: Mixed results across both types
- **Market Cap vs Stability**: USDT (largest) shows good stability
- **Backing Mechanism**: Fiat-backed (USDC/USDT) most stable

---

## 🛠️ **How to Use This Data**

### 📊 **For Research**
1. **Event Studies**: Use peg deviation data to identify policy impact
2. **Volatility Analysis**: Compare stability across different mechanisms
3. **Correlation Analysis**: Assess systemic risk in stablecoin markets
4. **Market Microstructure**: Use depth data for liquidity analysis

### 📈 **For Policy Analysis**
1. **Regulatory Impact**: Track stability before/after policy changes
2. **Market Stress**: Identify periods of systemic risk
3. **Competition Analysis**: Compare centralized vs decentralized models
4. **Risk Assessment**: Use volatility data for regulatory frameworks

### 🔍 **For Further Analysis**
1. **Time Series**: Analyze trends over the 2-year period
2. **Cross-Sectional**: Compare stablecoin characteristics
3. **Event Studies**: Link policy events to market responses
4. **Machine Learning**: Predict peg stability using market data

---

## 💡 **Next Steps**

### 🎯 **Immediate Actions**
1. **View Generated Plots**: Check `data/processed/` for visualizations
2. **Run Jupyter Notebooks**: Use `notebooks/00_exploration.ipynb`
3. **Deep Dive Analysis**: Explore specific stablecoin behaviors
4. **Policy Events**: Link market data to policy announcements

### 📊 **Advanced Analysis**
1. **GARCH Models**: Use volatility data for risk modeling
2. **VAR Analysis**: Study inter-stablecoin relationships
3. **Event Studies**: Measure policy impact on stability
4. **Machine Learning**: Predict peg breaks and market stress

### 🔬 **Research Extensions**
1. **Additional Stablecoins**: Add newer stablecoins to analysis
2. **Longer Time Series**: Extend data collection period
3. **Alternative Data**: Include on-chain metrics and sentiment
4. **Policy Database**: Build comprehensive policy event calendar

---

## 🎉 **Summary**

Your data reveals a **mature stablecoin ecosystem** with:
- **Clear stability hierarchy**: USDC/DAI most stable, LUSD most volatile
- **Systemic risk presence**: Moderate correlations suggest interconnected markets
- **Policy sensitivity**: Multiple depeg events show market stress periods
- **Diverse mechanisms**: Different backing models show varying performance

This comprehensive dataset provides a solid foundation for **policy research**, **risk assessment**, and **market analysis** in the stablecoin space! 🚀
