# Bitcoin Sentiment Index

A comprehensive dashboard that combines technical analysis with on-chain metrics and social sentiment to provide a sentiment index for Bitcoin investors.

## Features

- Market sentiment gauge based on multiple technical indicators
- Historical trends and analysis
- Price level targets and support/resistance levels
- On-chain metrics integration with Dune Analytics
- Reddit sentiment analysis from popular crypto subreddits
- ETF flow visualization
- Halving cycle context

## Setup

1. Clone this repository:
```
git clone https://github.com/atapinar/bitcoin-sentiment.git
cd bitcoin-sentiment
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
DUNE_API_KEY=your_dune_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=Bitcoin_Sentiment_Index_Bot
```

## API Setup (Optional)

By default, the dashboard will use simulated data. To display real data:

### Dune Analytics (On-chain metrics)

1. Create an account at [dune.com](https://dune.com)
2. Subscribe to a plan that includes API access
3. Get your API key from your profile settings
4. Add the API key to your `.env` file
5. Create your own queries in Dune or find existing ones
6. Update the query IDs in the code

### Reddit API (Social sentiment)

1. Create a Reddit account if you don't have one
2. Visit [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
3. Click "Create app" or "Create another app" at the bottom
4. Fill in the details:
   - Name: Bitcoin Sentiment Index
   - Type: Script
   - Description: App for analyzing Bitcoin sentiment on Reddit
   - About URL: (can be blank)
   - Redirect URI: http://localhost:8501
5. Copy your Client ID (under the app name) and Client Secret to your `.env` file

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
4. Changing which subreddits are analyzed for sentiment

## Data Sources

- Price data: Yahoo Finance API
- On-chain metrics: Dune Analytics (or simulated data if not configured)
- Social sentiment: Reddit API (or simulated data if not configured)
- ETF flows and Fear & Greed Index: Simulated (replace with real API as needed)

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice. Always do your own research before making investment decisions. 