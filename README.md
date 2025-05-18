# Bitcoin Sentiment Index

A comprehensive dashboard that combines technical analysis with on-chain metrics to provide a sentiment index for Bitcoin investors.

## Features

- Market sentiment gauge based on multiple technical indicators
- Historical trends and analysis
- Price level targets and support/resistance levels
- On-chain metrics integration with Dune Analytics
- ETF flow visualization
- Halving cycle context

## Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/bitcoin-sentiment-index.git
cd bitcoin-sentiment-index
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
DUNE_API_KEY=your_dune_api_key_here
```

## Dune Analytics Setup (Optional)

By default, the dashboard will use simulated on-chain data. To display real on-chain metrics:

1. Create an account at [dune.com](https://dune.com)
2. Subscribe to a plan that includes API access
3. Get your API key from your profile settings
4. Add the API key to your `.env` file
5. Create your own queries in Dune or find existing ones
6. Update the query IDs in the `get_btc_onchain_data()`, `get_btc_holder_distribution()`, and `get_btc_mining_data()` functions in the code

Example query parameters for Dune:

| Function | Data Required | 
|----------|---------------|
| active_addresses | Columns: date, active_addresses |
| transaction_volume | Columns: date, volume_usd |
| miner_revenue | Columns: date, revenue_usd |
| exchange_flows | Columns: date, net_flow_btc |
| holder_distribution | Columns: range, addresses |
| mining_data | Columns: date, hashrate_th_s, difficulty, block_time_seconds |

## Usage

Run the Streamlit app:
```
streamlit run crypto_market_sentiment_index.py
```

The dashboard will open in your browser at http://localhost:8501

## Customization

You can customize the dashboard by:

1. Adjusting the weights of different indicators in the sidebar
2. Modifying the visuals or adding new metrics
3. Adding your own Dune Analytics queries

## Data Sources

- Price data: Yahoo Finance API
- On-chain metrics: Dune Analytics (or simulated data if not configured)
- ETF flows and Fear & Greed Index: Simulated (replace with real API as needed)

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice. Always do your own research before making investment decisions. 