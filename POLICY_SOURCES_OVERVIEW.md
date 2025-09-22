# 📰 Policy News Sources & Input Overview

## 🎯 **Current Policy News Sources**

### 📊 **Primary Sources (Implemented)**

**1. Federal Reserve Official Sources:**
- **FOMC Minutes**: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- **Fed Speeches**: https://www.federalreserve.gov/newsevents/speech/
- **Press Releases**: https://www.federalreserve.gov/newsevents/pressreleases/
- **Testimonies**: https://www.federalreserve.gov/newsevents/testimony/

**2. Economic Data Sources:**
- **FRED (Federal Reserve Economic Data)**: Interest rates, inflation, employment
- **Treasury Data**: Yield curves, breakeven rates
- **Market Indicators**: VIX, DXY, economic indicators

### 🔍 **Policy Event Types Analyzed**

**Monetary Policy Events:**
- `fomc_meeting` - Federal Open Market Committee meetings
- `fomc_minutes` - Detailed meeting minutes (released 3 weeks after meetings)
- `fed_speech` - Speeches by Fed officials (Chair, Vice Chair, Governors)
- `rate_decision` - Federal Funds rate changes
- `qt_announcement` - Quantitative Tightening announcements

**Event Analysis Windows:**
- **Pre-event**: 5 trading days before policy announcement
- **Post-event**: 5 trading days after policy announcement
- **Estimation window**: 250 trading days for normal return calculation

---

## 📈 **Current Data Input Status**

### ✅ **What's Currently Working**

**Stablecoin Market Data (2023-09-22 to 2025-09-22):**
- **USDT**: Tether (largest stablecoin)
- **USDC**: USD Coin (Circle/Coinbase)
- **DAI**: MakerDAO algorithmic stablecoin
- **BUSD**: Binance USD (discontinued)
- **FRAX**: Fractional algorithmic stablecoin
- **LUSD**: Liquity USD (decentralized)
- **TUSD**: TrueUSD
- **USDP**: Pax Dollar (Paxos)

**Market Metrics Generated:**
- Daily prices and volumes
- Returns and volatility
- Peg deviations from $1.00
- Market depth and liquidity
- Rolling correlations between stablecoins

### ⚠️ **What's Currently Missing**

**Policy News Data:**
- **FOMC Minutes**: Scraping blocked by Fed website (403 errors)
- **Fed Speeches**: Scraping blocked by Fed website (403 errors)
- **Policy Events**: No policy event calendar currently populated

**Economic Data:**
- **FRED Data**: Requires API key (not configured)
- **Treasury Rates**: Requires API key (not configured)
- **Macro Indicators**: VIX, DXY, inflation data missing

---

## 🧠 **DeepSeek Sentiment Analysis Input**

### 📝 **Text Sources for Analysis**

**When Policy Data is Available:**
1. **FOMC Minutes Text**: Full meeting minutes content
2. **Speech Transcripts**: Complete speech text from Fed officials
3. **Press Release Text**: Policy announcement content
4. **Testimony Content**: Congressional testimony transcripts

**Sentiment Categories:**
- **Hawkish**: Tightening policy signals (rate hikes, QT, inflation focus)
- **Dovish**: Accommodative policy signals (rate cuts, QE, employment focus)
- **Neutral**: Balanced or data-dependent stance

### 🔍 **Analysis Output**

**For Each Policy Document:**
- **Sentiment Classification**: Hawkish/Dovish/Neutral
- **Confidence Score**: 0-1 scale
- **Key Phrases**: Supporting evidence extraction
- **Explanation**: Reasoning for classification
- **Timestamp**: When the policy was announced

---

## 🚀 **How to Add Policy News Input**

### 📊 **Option 1: Manual Policy Event Input**

Create a CSV file with policy events:

```csv
date,event_type,title,content,speaker,url
2024-01-31,FOMC_MEETING,FOMC January Meeting,The Committee decided to maintain the target range for the federal funds rate at 5.25 to 5.50 percent...,Jerome Powell,https://fed.gov/...
2024-02-21,FED_SPEECH,Monetary Policy Outlook,In my remarks today, I will discuss our current economic outlook...,Jerome Powell,https://fed.gov/...
```

### 📡 **Option 2: API-Based Data Sources**

**FRED API (Free):**
- Get API key: https://fred.stlouisfed.org/docs/api/api_key.html
- Add to `.env`: `FRED_API_KEY=your_key_here`

**Alternative Policy Data Sources:**
- **Fed API**: https://api.federalreserve.gov/
- **Economic Calendar APIs**: Investing.com, Trading Economics
- **News APIs**: NewsAPI, Alpha Vantage News

### 🔧 **Option 3: Web Scraping Enhancement**

**Improve Fed Website Scraping:**
- Add better headers and user agents
- Implement proxy rotation
- Add retry logic with exponential backoff
- Use Selenium for JavaScript-heavy pages

---

## 💡 **Recommended Next Steps**

### 🎯 **Immediate Actions**

1. **Add FRED API Key**: Get free API key for economic data
2. **Create Sample Policy Events**: Add manual policy event CSV
3. **Test DeepSeek Analysis**: Run sentiment analysis on sample policy text

### 📈 **Enhanced Analysis**

1. **Policy Event Calendar**: Build comprehensive event database
2. **Real-time Updates**: Set up automated policy data collection
3. **Multi-source Validation**: Cross-reference multiple policy sources
4. **Historical Analysis**: Extend analysis to longer time periods

### 🔬 **Research Extensions**

1. **International Policy**: Add ECB, BoE, BoJ policy events
2. **Crypto-specific Policy**: SEC announcements, regulatory changes
3. **Market Commentary**: Analyst reports, financial media sentiment
4. **Social Media**: Twitter/X sentiment from key policymakers

---

## 📊 **Current Analysis Capabilities**

**With Existing Data:**
- ✅ Stablecoin market behavior analysis
- ✅ Volatility and correlation patterns
- ✅ Peg deviation analysis
- ✅ Market microstructure metrics

**With Policy Data Added:**
- ✅ Event study analysis (CAR, BHAR)
- ✅ Policy sentiment impact on stablecoins
- ✅ Volatility regime changes around policy events
- ✅ Cross-stablecoin policy transmission effects

**Your repository is ready to analyze policy impact on stablecoins - it just needs policy event data as input!** 🚀
