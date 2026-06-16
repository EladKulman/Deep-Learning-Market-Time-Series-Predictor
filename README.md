# Deep Learning Time-Series Forecasting Project

This project builds a data ingestion layer to forecast short-term Nasdaq-100 / QQQ returns using various features including market price data, macro/market indicators, and alternative data.

## Directory Structure

- `data/raw/`: Contains raw downloaded datasets from various sources.
- `data/processed/`: Contains cleaned and normalized CSV files, combined by daily frequency.
- `data/logs/`: Logs for data ingestion scripts.
- `scripts/`: Contains Python scripts for fetching and processing data.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the `.env.example` file to `.env` and fill in your API keys (e.g., FRED API key).
   ```bash
   cp .env.example .env
   ```

## Ingestion Scripts

### Market Prices
```bash
python scripts/fetch_qqq_prices.py --start 2006-01-01
```

### Macro Data
```bash
python scripts/fetch_fred_macro.py --start 2006-01-01
```
**Warning:** The FRED script forward-fills lower-frequency data (like monthly CPI) to a daily frequency. To avoid data leakage in model training, ensure you only use data that was actually published on or before the current date. Future data is not used for creating daily features.

### VIX
```bash
python scripts/fetch_vix_cboe.py --start 2006-01-01
```

### GDELT News
```bash
python scripts/fetch_gdelt_news.py --start 2023-01-01 --end 2023-01-31
```
**Note:** GDELT heavily rate-limits public requests. It is recommended to query small batches of dates at a time.

### SEC Filings
```bash
python scripts/fetch_sec_filings.py --start 2023-01-01
```

### FOMC Statements
```bash
python scripts/fetch_fomc_statements.py --start 2023-01-01
```

## Output Data Structure

The processed output files are saved in `data/processed/` and share a common `date` column (in `YYYY-MM-DD` format) so they can easily be joined together.

### A. QQQ Prices (`data/processed/qqq_ohlcv_daily.csv`) 🟢 Success
This file contains the core market data for the Nasdaq-100 ETF, aligned to US trading days.
- **Base columns**: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`
- **Features**:
  - `return_1d`: The 1-day percentage change (using adjusted close).
  - `return_5d`: The 5-day percentage change.
  - `log_return_1d`: The natural logarithm of the 1-day return.
  - `volatility_20d`: A rolling 20-day standard deviation of the log returns.
  - `volume_change_1d`: The percentage change in trading volume from the previous day.

### B. FRED Macro Indicators (`data/processed/fred_macro_daily.csv`) 🟢 Success
This file contains macroeconomic data. Since some macro data (like CPI) is only reported monthly, this script automatically forward-fills the data to a daily frequency so it perfectly aligns with the daily stock prices.
- **Base columns**: `date`
- **Features**:
  - `dff`: Effective Federal Funds Rate.
  - `dgs10`: 10-Year Treasury Constant Maturity Rate.
  - `dgs2`: 2-Year Treasury Constant Maturity Rate.
  - `t10y2y`: The 10-Year minus 2-Year Treasury yield spread (a common recession indicator).
  - `cpi`: The Consumer Price Index (forward-filled monthly data).
  - `cpi_yoy`: Year-over-year percentage change in CPI (computed *before* forward filling to prevent calculation errors).
  - `vix_fred`: The VIX as reported by FRED (acts as a backup).

### C. CBOE VIX (`data/processed/vix_daily.csv`) 🟢 Success
This file contains the official CBOE Volatility Index, which tracks market expectations for volatility over the next 30 days.
- **Base columns**: `date`
- **Features**: 
  - `vix_open`, `vix_high`, `vix_low`, `vix_close`

### D. GDELT News Features (`data/processed/gdelt_topic_daily.csv`) 🔴 Rate Limited
This file contains daily aggregations of news articles matching specific tech and macro themes.
- **Base columns**: `date`
- **Features**:
  - `ai_news_count`, `ai_avg_tone`
  - `semiconductor_news_count`, `semiconductor_avg_tone`
  - `fed_news_count`, `fed_avg_tone`
  - `inflation_news_count`, `inflation_avg_tone`
  - `big_tech_earnings_count`, `big_tech_earnings_avg_tone`
  - `recession_news_count`, `recession_avg_tone`
*(Note: `_avg_tone` columns are currently placeholders for future sentiment analysis).*

### E. SEC Filings (`data/processed/sec_filings_daily.csv`) 🟢 Success
This file tracks major corporate filing events (10-K, 10-Q, 8-K) for the top Nasdaq companies.
- **Base columns**: `date`
- **Features**:
  - `sec_10k_count`: Number of 10-K filings across tracked companies.
  - `sec_10q_count`: Number of 10-Q filings.
  - `sec_8k_count`: Number of 8-K filings.
  - `sec_total_filings`: Sum of the above.
  - `{ticker}_filing_event`: Binary flag (1 or 0) if a specific company (e.g., `nvda_filing_event`) filed a document on this date.

### F. FOMC Statements (`data/processed/fomc_events_daily.csv`) 🟢 Success
This file parses Federal Reserve monetary policy statements.
- **Base columns**: `date`
- **Features**:
  - `is_fomc_day`: Binary flag (1 if an FOMC statement was released).
  - `days_since_fomc`: Counter that resets to 0 on statement days.
  - `fomc_statement_length`: Character length of the statement.
  - `fomc_hawkish_keyword_count`: Frequency of hawkish terms (e.g., "tightening", "inflation").
  - `fomc_dovish_keyword_count`: Frequency of dovish terms (e.g., "easing", "rate cut").
