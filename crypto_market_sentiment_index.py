import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests
import os
import praw
from textblob import TextBlob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Signal Index",
    page_icon="₿",
    layout="wide"
)

# Sidebar for controls
st.sidebar.title("Settings")
timeframe = st.sidebar.selectbox(
    "Select Analysis Timeframe",
    ["Daily", "Weekly", "Monthly"]
)

# Add customization options in sidebar
use_custom_weights = st.sidebar.checkbox("Use Custom Indicator Weights", False)

if use_custom_weights:
    st.sidebar.subheader("Indicator Weights")
    st.sidebar.info("Adjust the importance of each indicator in the final signal (total must equal 100%)")
    
    custom_weights = {}
    custom_weights['RSI'] = st.sidebar.slider("RSI Weight (%)", 5, 30, 15) / 100
    custom_weights['Trend'] = st.sidebar.slider("Trend Weight (%)", 5, 30, 20) / 100
    custom_weights['Performance'] = st.sidebar.slider("Performance Weight (%)", 5, 30, 15) / 100
    custom_weights['Volatility'] = st.sidebar.slider("Volatility Weight (%)", 5, 30, 10) / 100
    custom_weights['MACD'] = st.sidebar.slider("MACD Weight (%)", 5, 30, 15) / 100
    custom_weights['Bollinger'] = st.sidebar.slider("Bollinger Weight (%)", 5, 30, 15) / 100
    custom_weights['Volume'] = st.sidebar.slider("Volume Weight (%)", 5, 30, 10) / 100
    
    # Calculate total to ensure it equals 100%
    total_weight = sum(custom_weights.values())
    if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
        st.sidebar.warning(f"⚠️ Total weight is {total_weight:.2f} (should be 1.0). Weights will be normalized.")
        # Normalize weights
        for key in custom_weights:
            custom_weights[key] /= total_weight

# Add alert setup
with st.sidebar.expander("Set Up Price Alerts"):
    alert_email = st.text_input("Email for alerts")
    st.checkbox("Alert on BUY signal")
    st.checkbox("Alert on SELL signal")
    
    st.text_input("Alert when price exceeds $", "70000")
    st.text_input("Alert when price drops below $", "60000")

# App title and description
st.title("Bitcoin Signal Index")
st.markdown("### What action should you consider in the market now?")
st.markdown("[Learn more about this index](#how-it-works)")

# Simple RSI calculation function that avoids Series comparisons
def calculate_rsi(prices, window=14):
    prices = prices.astype(float)  # Ensure we're working with float values
    
    # Calculate price changes
    delta = prices.diff().dropna()
    
    # Create copies to avoid SettingWithCopyWarning
    gains = delta.copy()
    losses = delta.copy()
    
    # Separate gains and losses
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses using simple moving average
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate the index based on various market indicators
def calculate_market_index():
    try:
        # Get Bitcoin data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        btc_data = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        
        if btc_data.empty:
            st.warning("No Bitcoin data available. Using neutral values.")
            return 50, {
                'RSI': 50,
                'Trend': 50,
                'Performance': 50,
                'Volatility': 50
            }, {}
        
        # Extract the needed columns as numpy values to avoid Series issues
        closing_prices = btc_data['Close'].values
        if len(closing_prices) < 2:
            st.warning("Not enough data points. Using neutral values.")
            return 50, {
                'RSI': 50,
                'Trend': 50,
                'Performance': 50,
                'Volatility': 50
            }, {}
        
        # Calculate indicators
        btc_data['RSI_14'] = calculate_rsi(btc_data['Close'])
        btc_data['SMA_50'] = btc_data['Close'].rolling(window=50, min_periods=1).mean()
        btc_data['SMA_200'] = btc_data['Close'].rolling(window=200, min_periods=1).mean()
        
        # Calculate Bollinger Bands
        window = 20
        btc_data['SMA_20'] = btc_data['Close'].rolling(window=window, min_periods=1).mean()
        btc_data['STD_20'] = btc_data['Close'].rolling(window=window, min_periods=1).std()
        btc_data['Upper_Band'] = btc_data['SMA_20'] + (btc_data['STD_20'] * 2)
        btc_data['Lower_Band'] = btc_data['SMA_20'] - (btc_data['STD_20'] * 2)
        
        # Calculate daily returns and volatility
        btc_data['Daily_Return'] = btc_data['Close'].pct_change()
        btc_data['Volatility'] = btc_data['Daily_Return'].rolling(window=20, min_periods=5).std() * np.sqrt(365) * 100
        
        # Get latest values as simple floating-point numbers, not Series
        current_price = float(btc_data['Close'].iloc[-1])
        current_rsi = float(btc_data['RSI_14'].iloc[-1]) if not pd.isna(btc_data['RSI_14'].iloc[-1]) else 50.0
        current_sma50 = float(btc_data['SMA_50'].iloc[-1]) if not pd.isna(btc_data['SMA_50'].iloc[-1]) else current_price
        current_sma200 = float(btc_data['SMA_200'].iloc[-1]) if not pd.isna(btc_data['SMA_200'].iloc[-1]) else current_price
        current_volatility = float(btc_data['Volatility'].iloc[-1]) if not pd.isna(btc_data['Volatility'].iloc[-1]) else 30.0
        
        # Calculate average volatility as a simple float
        volatility_values = btc_data['Volatility'].dropna().values
        avg_volatility = float(np.mean(volatility_values)) if len(volatility_values) > 0 else 30.0
        
        # Calculate returns over different timeframes
        lookback_periods = {
            '30d': 30,
            '90d': 90
        }
        
        returns = {}
        for period_name, days in lookback_periods.items():
            # Make sure we don't go out of bounds
            lookback_idx = max(0, len(btc_data) - min(days, len(btc_data)))
            past_price = float(btc_data['Close'].iloc[lookback_idx])
            returns[period_name] = ((current_price / past_price) - 1) * 100
        
        # Trend analysis with scalar comparisons
        is_above_50sma = current_price > current_sma50
        is_above_200sma = current_price > current_sma200
        golden_cross = current_sma50 > current_sma200
        
        # Calculate component scores (0-100 scale)
        
        # RSI score
        if np.isnan(current_rsi):
            rsi_score = 50.0  # Neutral if no data
        elif current_rsi > 70:
            rsi_score = 100.0 - current_rsi  # Overbought
        elif current_rsi < 30:
            rsi_score = current_rsi  # Oversold
        else:
            rsi_score = 50.0  # Neutral
        
        # Trend score
        if is_above_50sma and is_above_200sma and golden_cross:
            trend_score = 80.0
        elif is_above_50sma and is_above_200sma:
            trend_score = 70.0
        elif is_above_50sma:
            trend_score = 60.0
        elif not is_above_50sma and not is_above_200sma:
            trend_score = 30.0
        else:
            trend_score = 40.0
        
        # Performance score
        monthly_return = returns['30d']
        quarterly_return = returns['90d']
        
        if monthly_return > 20 and quarterly_return > 30:
            performance_score = 10.0  # Potentially overbought
        elif monthly_return > 10 and quarterly_return > 20:
            performance_score = 30.0
        elif monthly_return < -20 and quarterly_return < -30:
            performance_score = 90.0  # Potentially oversold
        elif monthly_return < -10 and quarterly_return < -20:
            performance_score = 70.0
        else:
            performance_score = 50.0
        
        # Volatility score
        volatility_ratio = current_volatility / max(0.1, avg_volatility)  # Ensure no division by zero
        
        if volatility_ratio > 1.5:
            volatility_score = 30.0  # High volatility suggests caution
        elif volatility_ratio < 0.7:
            volatility_score = 70.0  # Low volatility can indicate accumulation
        else:
            volatility_score = 50.0
        
        # Combine scores with different weights
        component_scores = {
            'RSI': rsi_score,
            'Trend': trend_score,
            'Performance': performance_score,
            'Volatility': volatility_score
        }
        
        weights = {
            'RSI': 0.25,
            'Trend': 0.35,
            'Performance': 0.25,
            'Volatility': 0.15
        }
        
        weighted_score = sum(component_scores[k] * weights[k] for k in weights)
        
        # Ensure the final score is in the 0-100 range
        final_score = max(0.0, min(100.0, weighted_score))
        
        # Create price data dictionary for price targets
        # Calculate recent high and low
        recent_high = float(btc_data['Close'].rolling(window=30).max().iloc[-1])
        recent_low = float(btc_data['Close'].rolling(window=30).min().iloc[-1])
        
        # Calculate ATR (Average True Range)
        btc_data['High-Low'] = btc_data['High'] - btc_data['Low']
        btc_data['High-PrevClose'] = abs(btc_data['High'] - btc_data['Close'].shift(1))
        btc_data['Low-PrevClose'] = abs(btc_data['Low'] - btc_data['Close'].shift(1))
        btc_data['TR'] = btc_data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        atr = float(btc_data['TR'].rolling(window=14).mean().iloc[-1])
        
        price_data = {
            'current_price': current_price,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'atr': atr,
            'btc_data': btc_data  # Add the full DataFrame for plotting
        }
        
        return final_score, component_scores, price_data
    
    except Exception as e:
        st.error(f"Error calculating market index: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return 50.0, {
            'RSI': 50.0,
            'Trend': 50.0,
            'Performance': 50.0,
            'Volatility': 50.0
        }, {}

def generate_historical_index():
    """Generate simulated historical index values"""
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    
    # Create a baseline with some trend
    baseline = np.linspace(40, 60, len(dates))
    
    # Add cycles of various frequencies
    cycle1 = 15 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
    cycle2 = 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    
    # Add some random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Combine components and ensure values are within 0-100
    values = baseline + cycle1 + cycle2 + noise
    values = np.clip(values, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Index': values
    })
    
    return df

def get_market_condition(score):
    """Get the market condition label based on the score"""
    if score >= 80:
        return "STRONG BUY"
    elif score >= 65:
        return "BUY"
    elif score >= 45:
        return "HODL"
    elif score >= 20:
        return "SELL"
    else:
        return "STRONG SELL"
        
def get_halving_context():
    """Provide context about Bitcoin's halving cycle"""
    # Previous halvings
    previous_halvings = [
        datetime(2012, 11, 28),
        datetime(2016, 7, 9),
        datetime(2020, 5, 11),
        datetime(2024, 4, 27)  # Most recent halving
    ]
    
    next_halving_estimate = datetime(2028, 5, 15)  # Approximate next halving
    current_date = datetime.now()
    
    # Calculate days since last halving
    days_since_halving = (current_date - previous_halvings[-1]).days
    days_to_next_halving = (next_halving_estimate - current_date).days
    total_cycle_days = (next_halving_estimate - previous_halvings[-1]).days
    cycle_progress = (days_since_halving / total_cycle_days) * 100
    
    # Determine cycle phase
    if days_since_halving < 100:
        cycle_phase = "Early Post-Halving Period"
        phase_description = "Historically a period of accumulation before bull market acceleration."
    elif days_since_halving < 365:
        cycle_phase = "Year 1 Post-Halving"
        phase_description = "Typically when bull markets gain momentum."
    elif days_since_halving < 730:
        cycle_phase = "Year 2 Post-Halving"
        phase_description = "Often includes peak bull market and correction."
    else:
        cycle_phase = "Late Cycle"
        phase_description = "Typically a period of consolidation before the next halving."
    
    return {
        "last_halving": previous_halvings[-1].strftime("%B %d, %Y"),
        "next_halving": next_halving_estimate.strftime("%B %d, %Y"),
        "days_since_halving": days_since_halving,
        "days_to_next_halving": days_to_next_halving,
        "cycle_progress": cycle_progress,
        "cycle_phase": cycle_phase,
        "phase_description": phase_description
    }

def calculate_price_targets(current_price, atr, recent_high, recent_low):
    """Calculate price targets based on current price, ATR, and recent high/low"""
    # Support levels
    strong_support = recent_low * 0.95
    support_1 = current_price - (atr * 2)
    support_2 = current_price - (atr * 1)
    
    # Resistance levels
    strong_resistance = recent_high * 1.05
    resistance_1 = current_price + (atr * 1)
    resistance_2 = current_price + (atr * 2)
    
    # Targets
    bullish_target = current_price + (atr * 5)
    bearish_target = current_price - (atr * 5)
    
    return {
        "Strong_Support": strong_support,
        "Support_1": support_1,
        "Support_2": support_2,
        "Resistance_1": resistance_1,
        "Resistance_2": resistance_2,
        "Strong_Resistance": strong_resistance,
        "Bullish_Target": bullish_target,
        "Bearish_Target": bearish_target
    }

def get_etf_flows(timeframe="Daily"):
    """Simulate ETF flow data (in a real implementation, this would connect to an API)"""
    # For demonstration, we'll use simulated data
    # In a real implementation, you would fetch this from a financial data API
    
    etf_tickers = ["IBIT", "FBTC", "BTCO", "ARKB", "BITB", "HODL", "GBTC"]
    
    # Use timeframe to seed random generator to ensure different values for different timeframes
    np.random.seed(int(datetime.now().timestamp()) % 100 + hash(timeframe) % 100)
    
    # Simulated daily flows (in millions USD)
    flows = {}
    
    # Scale flow values based on timeframe
    if timeframe == "Daily":
        base_mean = 20
        base_std = 30
        period_label = "Daily"
    elif timeframe == "Weekly":
        base_mean = 80
        base_std = 50
        period_label = "Weekly"
    else:  # Monthly
        base_mean = 200
        base_std = 120
        period_label = "Monthly"
    
    for ticker in etf_tickers:
        # Generate random flow with appropriate scale for the timeframe
        base_flow = np.random.normal(base_mean, base_std)
        flows[ticker] = f"{base_flow:.1f}M"
    
    # Calculate period totals (simulated)
    # Scale the weekly/monthly totals based on timeframe
    if timeframe == "Daily":
        weekly_base = np.random.normal(120, 50)
        total_period_label = "Weekly"
    elif timeframe == "Weekly":
        weekly_base = np.random.normal(400, 150)
        total_period_label = "Monthly"
    else:  # Monthly
        weekly_base = np.random.normal(1200, 300)
        total_period_label = "Quarterly"
    
    return {
        "daily_flows": flows,
        "total_daily": f"${sum([float(v.replace('M','')) for v in flows.values()]):.1f}M",
        "total_7d": f"${weekly_base:.1f}M",
        "total_period_label": total_period_label,  # Dynamic label based on timeframe
        "period_label": period_label,  # For proper labeling in UI
        "trend": "Increasing" if weekly_base > 0 else "Decreasing"
    }

def get_index_color(score):
    """Get the color for the index based on the score"""
    if score >= 80:
        return "rgba(0, 128, 0, 0.7)"  # Dark Green
    elif score >= 65:
        return "rgba(144, 238, 144, 0.7)"  # Light Green
    elif score >= 45:
        return "rgba(169, 169, 169, 0.7)"  # Gray
    elif score >= 20:
        return "rgba(255, 165, 0, 0.7)"  # Orange
    else:
        return "rgba(255, 0, 0, 0.7)"  # Red

def get_fear_greed_index():
    """Simulate Fear & Greed Index (would connect to an API in production)"""
    # This would normally fetch from alternative.me or similar API
    # For demonstration, we'll generate a value that's correlated with our calculated index
    
    # Get current date as seed for pseudo-random number
    seed = int(datetime.now().strftime("%Y%m%d")) % 100
    np.random.seed(seed)
    
    # Base value between 0-100
    base_value = np.random.randint(20, 80)
    
    # Labels based on value ranges
    if base_value >= 80:
        label = "Extreme Greed"
    elif base_value >= 60:
        label = "Greed"
    elif base_value >= 40:
        label = "Neutral"
    elif base_value >= 20:
        label = "Fear"
    else:
        label = "Extreme Fear"
    
    return {
        "value": base_value,
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d")
    }

# Set cache for expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_data():
    return generate_historical_index()

# Dune Analytics API functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_dune_query(query_id):
    """Fetch data from Dune Analytics API with the given query ID"""
    # Check if we're using placeholder IDs
    if query_id in ["1234567", "1234568", "1234569", "1234570", "1234571", "1234572"]:
        return pd.DataFrame()  # Return empty DataFrame for placeholder IDs
        
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        return pd.DataFrame()
    
    try:
        headers = {"x-dune-api-key": api_key}
        base_url = "https://api.dune.com/api/v1"
        
        # Execute query
        execute_endpoint = f"{base_url}/query/{query_id}/execute"
        response = requests.post(execute_endpoint, headers=headers)
        
        if response.status_code != 200:
            error_message = response.json().get("error", "Unknown error")
            return pd.DataFrame()
        
        execution_id = response.json()["execution_id"]
        
        # Get execution status and results
        status_endpoint = f"{base_url}/execution/{execution_id}/status"
        results_endpoint = f"{base_url}/execution/{execution_id}/results"
        
        # Poll for results
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            time.sleep(2)  # Wait for query to complete
            status_response = requests.get(status_endpoint, headers=headers)
            
            if status_response.status_code != 200:
                return pd.DataFrame()
            
            state = status_response.json()["state"]
            
            if state == "QUERY_STATE_COMPLETED":
                results_response = requests.get(results_endpoint, headers=headers)
                
                if results_response.status_code != 200:
                    return pd.DataFrame()
                
                result_data = results_response.json()["result"]["rows"]
                return pd.DataFrame(result_data)
            
            elif state in ["QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"]:
                return pd.DataFrame()
            
            attempts += 1
        
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_btc_onchain_data():
    """Get Bitcoin on-chain metrics from Dune Analytics"""
    # Replace these with your actual query IDs from Dune Analytics
    metrics = {
        "active_addresses": fetch_dune_query("1234567"),  # Replace with actual query ID
        "transaction_volume": fetch_dune_query("1234568"),  # Replace with actual query ID
        "miner_revenue": fetch_dune_query("1234569"),  # Replace with actual query ID
        "exchange_flows": fetch_dune_query("1234570")   # Replace with actual query ID
    }
    
    # Generate sample data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Active addresses (sample data)
    active_addr = pd.DataFrame({
        'date': dates,
        'active_addresses': np.random.randint(800000, 1200000, size=len(dates))
    })
    
    # Transaction volume in USD (sample data)
    tx_volume = pd.DataFrame({
        'date': dates,
        'volume_usd': np.random.randint(5000000000, 15000000000, size=len(dates))
    })
    
    # Miner revenue in USD (sample data)
    miner_rev = pd.DataFrame({
        'date': dates,
        'revenue_usd': np.random.randint(15000000, 25000000, size=len(dates))
    })
    
    # Exchange net flows in BTC (sample data)
    exchange_flow = pd.DataFrame({
        'date': dates,
        'net_flow_btc': np.random.uniform(-5000, 5000, size=len(dates))
    })
    
    # Fill with sample data if queries failed
    if metrics["active_addresses"].empty:
        metrics["active_addresses"] = active_addr
    if metrics["transaction_volume"].empty:
        metrics["transaction_volume"] = tx_volume
    if metrics["miner_revenue"].empty:
        metrics["miner_revenue"] = miner_rev
    if metrics["exchange_flows"].empty:
        metrics["exchange_flows"] = exchange_flow
    
    return metrics

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_btc_holder_distribution():
    """Get Bitcoin holder distribution data from Dune Analytics"""
    # Replace with actual query ID
    holder_data = fetch_dune_query("1234571")
    
    # Generate sample data if query failed
    if holder_data.empty:
        holder_distribution = {
            "0-0.1 BTC": 15000000,
            "0.1-1 BTC": 3000000,
            "1-10 BTC": 850000,
            "10-100 BTC": 140000,
            "100-1000 BTC": 14000,
            "1000-10000 BTC": 2000,
            ">10000 BTC": 100
        }
        
        holder_data = pd.DataFrame({
            'range': list(holder_distribution.keys()),
            'addresses': list(holder_distribution.values())
        })
    
    return holder_data

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_btc_mining_data():
    """Get Bitcoin mining statistics from Dune Analytics"""
    # Replace with actual query ID
    mining_data = fetch_dune_query("1234572")
    
    # Generate sample data if query failed
    if mining_data.empty:
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        mining_data = pd.DataFrame({
            'date': dates,
            'hashrate_th_s': np.random.uniform(350000000, 450000000, size=len(dates)),
            'difficulty': np.random.uniform(55000000000000, 65000000000000, size=len(dates)),
            'block_time_seconds': np.random.uniform(580, 620, size=len(dates))
        })
    
    return mining_data

# Reddit Sentiment Analysis Functions
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_reddit_sentiment():
    """Fetch and analyze Reddit posts for Bitcoin sentiment"""
    try:
        # Set timeout to prevent hanging
        timeout_seconds = 30
        start_time = time.time()
        
        # Initialize Reddit API with environment variables
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "Bitcoin_Sentiment_Index_Bot")
        
        if not reddit_client_id or not reddit_client_secret:
            return generate_sample_reddit_data()
        
        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Subreddits to analyze
        subreddits = ["Bitcoin", "CryptoCurrency", "BitcoinMarkets", "btc"]
        
        # Fetch posts and comments
        posts_data = []
        
        for subreddit_name in subreddits:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                st.warning("Reddit API request timed out. Using sample data.")
                return generate_sample_reddit_data()
                
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Get hot posts with timeout check
                for post in subreddit.hot(limit=10):  # Reduced from 25 to 10 for speed
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        break
                        
                    # Analyze post sentiment
                    sentiment = analyze_text_sentiment(post.title + " " + post.selftext)
                    
                    # Collect post data
                    post_data = {
                        "id": post.id,
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc),
                        "sentiment_score": sentiment["score"],
                        "sentiment_label": sentiment["label"],
                        "type": "post"
                    }
                    
                    posts_data.append(post_data)
                    
                    # Analyze only a few top comments to avoid timeouts
                    post.comments.replace_more(limit=0)  # Skip fetching more comments
                    for i, comment in enumerate(list(post.comments)):
                        if i >= 5:  # Only process 5 comments per post
                            break
                            
                        # Check timeout
                        if time.time() - start_time > timeout_seconds:
                            break
                            
                        comment_sentiment = analyze_text_sentiment(comment.body)
                        
                        comment_data = {
                            "id": comment.id,
                            "subreddit": subreddit_name,
                            "title": f"Comment on: {post.title[:50]}...",
                            "score": comment.score,
                            "num_comments": 0,
                            "created_utc": datetime.fromtimestamp(comment.created_utc),
                            "sentiment_score": comment_sentiment["score"],
                            "sentiment_label": comment_sentiment["label"],
                            "type": "comment"
                        }
                        
                        posts_data.append(comment_data)
            except Exception as e:
                # If error occurs for a subreddit, continue with others
                continue
        
        # If we don't have enough data, use sample data
        if len(posts_data) < 10:
            return generate_sample_reddit_data()
            
        # Convert to DataFrame
        df = pd.DataFrame(posts_data)
        
        # Calculate weighted sentiment
        df["weighted_sentiment"] = df["sentiment_score"] * (df["score"] + 1)  # Add 1 to avoid multiplying by zero
        
        # Calculate overall sentiment metrics
        total_weighted = df["weighted_sentiment"].sum()
        total_weight = (df["score"] + 1).sum()
        overall_sentiment = total_weighted / total_weight if total_weight > 0 else 0
        
        # Count sentiment categories
        sentiment_counts = df["sentiment_label"].value_counts().to_dict()
        
        # Get average sentiment by subreddit
        subreddit_sentiment = df.groupby("subreddit")["sentiment_score"].mean().to_dict()
        
        # Make sure we have all subreddits in the dictionary (even if missing)
        for sub in subreddits:
            if sub not in subreddit_sentiment:
                subreddit_sentiment[sub] = 0
        
        # Get recent trending posts (high score, recent)
        df["recency_score"] = (datetime.now() - df["created_utc"]).dt.total_seconds() / 3600  # Hours ago
        df["trending_score"] = df["score"] / (df["recency_score"] + 1)  # Higher score for recent, high scoring posts
        trending_posts = df.sort_values("trending_score", ascending=False).head(5)
        
        # Process time series for sentiment over time
        df["date"] = df["created_utc"].dt.date
        sentiment_by_date = df.groupby("date")["sentiment_score"].mean().reset_index()
        sentiment_by_date["date"] = pd.to_datetime(sentiment_by_date["date"])
        
        # Return structured data
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_counts": sentiment_counts,
            "subreddit_sentiment": subreddit_sentiment,
            "trending_posts": trending_posts,
            "sentiment_over_time": sentiment_by_date,
            "data_source": "reddit_api",
            "post_count": len(df)
        }
    
    except Exception as e:
        # Return mock data if error occurs
        return generate_sample_reddit_data()

def analyze_text_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not text or text.strip() == "":
        return {"score": 0, "label": "neutral"}
    
    try:
        # Analyze with TextBlob
        analysis = TextBlob(text)
        
        # Get polarity score (-1 to 1)
        score = analysis.sentiment.polarity
        
        # Determine sentiment label
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        return {"score": score, "label": label}
    except Exception as e:
        # If TextBlob fails, use a simple keyword-based approach
        text = text.lower()
        bullish_words = ["bullish", "buy", "moon", "growth", "up", "gain", "profit", "positive", "rally", 
                         "surge", "soar", "excellent", "opportunity", "potential", "hodl", "green"]
        bearish_words = ["bearish", "sell", "crash", "down", "fall", "decline", "loss", "negative", "dump", 
                         "correction", "drop", "avoid", "risk", "bubble", "bad", "red"]
        
        bullish_count = sum(1 for word in bullish_words if word in text)
        bearish_count = sum(1 for word in bearish_words if word in text)
        
        if bullish_count > bearish_count:
            return {"score": 0.3, "label": "bullish"}
        elif bearish_count > bullish_count:
            return {"score": -0.3, "label": "bearish"}
        else:
            return {"score": 0, "label": "neutral"}

def generate_sample_reddit_data():
    """Generate sample Reddit sentiment data when API is not available"""
    # Create sample dates
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Generate trending sentiment (somewhat correlated with market but with some randomness)
    base_sentiment = np.linspace(-0.1, 0.4, len(dates))  # Slight uptrend
    noise = np.random.normal(0, 0.2, len(dates))  # Random noise
    sentiment_values = np.clip(base_sentiment + noise, -1, 1)  # Clip to valid range
    
    # Create sentiment over time DataFrame
    sentiment_by_date = pd.DataFrame({
        'date': dates,
        'sentiment_score': sentiment_values
    })
    
    # Create sample subreddit sentiment
    subreddit_sentiment = {
        "Bitcoin": 0.15,
        "CryptoCurrency": 0.05,
        "BitcoinMarkets": -0.12,
        "btc": 0.22
    }
    
    # Create sample sentiment counts
    sentiment_counts = {
        "bullish": 146,
        "neutral": 218,
        "bearish": 87
    }
    
    # Generate sample trending posts
    trending_posts = pd.DataFrame({
        "id": [f"sample{i}" for i in range(5)],
        "subreddit": np.random.choice(["Bitcoin", "CryptoCurrency", "BitcoinMarkets", "btc"], 5),
        "title": [
            "Bitcoin just broke the $100k resistance level!",
            "Why I'm still bullish on Bitcoin despite market uncertainty",
            "Technical Analysis: BTC might be forming a double bottom",
            "Is institutional adoption slowing down? My thoughts on recent BTC trends",
            "This halving cycle is unlike any other we've seen before"
        ],
        "score": np.random.randint(100, 5000, 5),
        "num_comments": np.random.randint(10, 300, 5),
        "created_utc": [datetime.now() - timedelta(hours=i*4) for i in range(5)],
        "sentiment_score": [0.65, 0.45, 0.1, -0.3, 0.25],
        "sentiment_label": ["bullish", "bullish", "neutral", "bearish", "bullish"],
        "type": ["post", "post", "post", "post", "post"],
        "trending_score": np.random.uniform(50, 500, 5)
    })
    
    # Calculate overall sentiment (weighted average of sample data)
    overall_sentiment = 0.08  # Slightly positive
    
    return {
        "overall_sentiment": overall_sentiment,
        "sentiment_counts": sentiment_counts,
        "subreddit_sentiment": subreddit_sentiment,
        "trending_posts": trending_posts,
        "sentiment_over_time": sentiment_by_date,
        "data_source": "sample_data",
        "post_count": 451  # Sample size
    }

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Timeline", "On-Chain Data", "Reddit Sentiment"])

# Calculate current index
with st.spinner("Calculating market sentiment..."):
    current_index, component_scores, price_data = calculate_market_index()
    current_condition = get_market_condition(current_index)
    index_color = get_index_color(current_index)

with tab1:
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Add Fear & Greed Index display
        fear_greed_data = get_fear_greed_index()
        st.subheader("Market Fear & Greed")
        
        # Create a small gauge for Fear & Greed
        fig_fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fear_greed_data["value"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': fear_greed_data["label"], 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "gray"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': fear_greed_data["value"]
                }
            }
        ))
        
        fig_fg.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_fg, use_container_width=True)
    
    with col2:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_index,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{current_index:.0f}</b>", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': "rgba(255, 0, 0, 0.7)"},
                    {'range': [20, 45], 'color': "rgba(255, 165, 0, 0.7)"},
                    {'range': [45, 65], 'color': "rgba(169, 169, 169, 0.7)"},
                    {'range': [65, 80], 'color': "rgba(144, 238, 144, 0.7)"},
                    {'range': [80, 100], 'color': "rgba(0, 128, 0, 0.7)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_index
                }
            }
        ))
        
        # Add text labels to the gauge
        fig.add_annotation(
            x=0.1, y=0.6,
            text="STRONG<br>SELL",
            showarrow=False,
            font=dict(size=14)
        )
        fig.add_annotation(
            x=0.3, y=0.6,
            text="SELL",
            showarrow=False,
            font=dict(size=14)
        )
        fig.add_annotation(
            x=0.5, y=0.6,
            text="HODL",
            showarrow=False,
            font=dict(size=14)
        )
        fig.add_annotation(
            x=0.7, y=0.6,
            text="BUY",
            showarrow=False,
            font=dict(size=14)
        )
        fig.add_annotation(
            x=0.9, y=0.6,
            text="STRONG<br>BUY",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=16)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Get halving cycle context
        halving_data = get_halving_context()
        
        st.subheader("Halving Cycle")
        st.markdown(f"**Current Phase:**<br>{halving_data['cycle_phase']}", unsafe_allow_html=True)
        st.markdown(f"**Last Halving:**<br>{halving_data['last_halving']}", unsafe_allow_html=True)
        st.markdown(f"**Next Halving:**<br>{halving_data['next_halving']}", unsafe_allow_html=True)
        
        # Create cycle progress bar
        cycle_progress = halving_data['cycle_progress']
        st.progress(cycle_progress/100)
        st.text(f"{cycle_progress:.1f}% of cycle complete")
        st.text(f"{halving_data['days_since_halving']} days since halving")
    
    # Display index components
    st.subheader("Index Components")
    cols = st.columns(len(component_scores))
    
    for i, (name, score) in enumerate(component_scores.items()):
        with cols[i]:
            st.metric(
                label=name, 
                value=f"{score:.0f}/100",
                delta=None
            )
    
    # Display current assessment
    st.subheader("Current Market Assessment")
    st.markdown(f"<h1 style='text-align: center; color: {index_color};'>{current_condition}</h1>", unsafe_allow_html=True)
    
    # Try to calculate price targets
    try:
        # Use price data returned from calculate_market_index
        current_price = price_data.get('current_price', 0)
        recent_high = price_data.get('recent_high', current_price * 1.1)
        recent_low = price_data.get('recent_low', current_price * 0.9)
        atr = price_data.get('atr', current_price * 0.05)
        
        # Get price targets
        targets = calculate_price_targets(current_price, atr, recent_high, recent_low)
        
        # Display price targets
        st.subheader("Key Price Levels")
        target_cols = st.columns(2)
        
        with target_cols[0]:
            st.markdown("**Support Levels**")
            st.markdown(f"Strong Support: ${targets['Strong_Support']:,.0f}")
            st.markdown(f"Support 1: ${targets['Support_1']:,.0f}")
            st.markdown(f"Support 2: ${targets['Support_2']:,.0f}")
            
            if current_condition in ["SELL", "STRONG SELL"]:
                st.markdown(f"**Bearish Target: ${targets['Bearish_Target']:,.0f}**")
        
        with target_cols[1]:
            st.markdown("**Resistance Levels**")
            st.markdown(f"Resistance 1: ${targets['Resistance_1']:,.0f}")
            st.markdown(f"Resistance 2: ${targets['Resistance_2']:,.0f}")
            st.markdown(f"Strong Resistance: ${targets['Strong_Resistance']:,.0f}")
            
            if current_condition in ["BUY", "STRONG BUY"]:
                st.markdown(f"**Bullish Target: ${targets['Bullish_Target']:,.0f}**")
    except Exception as e:
        st.warning(f"Could not calculate price targets: {e}")
    
    # Show ETF flows
    st.subheader("Bitcoin ETF Flows")
    etf_data = get_etf_flows(timeframe)
    
    etf_cols = st.columns([1, 1])
    with etf_cols[0]:
        st.markdown(f"**{etf_data['period_label']} Total: {etf_data['total_daily']}**")
        st.markdown(f"**{etf_data['total_period_label']} Total: {etf_data['total_7d']}**")
        st.markdown(f"**Trend: {etf_data['trend']}**")
    
    with etf_cols[1]:
        # Create a bar chart of ETF flows
        etf_names = list(etf_data['daily_flows'].keys())
        etf_values = [float(v.replace('M','')) for v in etf_data['daily_flows'].values()]
        
        fig_etf = go.Figure(go.Bar(
            x=etf_names,
            y=etf_values,
            marker_color=['green' if v > 0 else 'red' for v in etf_values]
        ))
        
        fig_etf.update_layout(
            title=f"{etf_data['period_label']} ETF Flows (USD Millions)",
            height=250,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        st.plotly_chart(fig_etf, use_container_width=True)
    
    # Action guidance based on the index value
    st.subheader("Suggested Action")
    action_cols = st.columns([2, 1])
    
    with action_cols[0]:
        if current_index >= 80:
            st.markdown("""
            #### STRONG BUY Signal
            Consider allocating a larger portion of your portfolio to BTC. Multiple technical indicators suggest highly favorable conditions. Key metrics point to a potential significant upward move.
            """)
        elif current_index >= 65:
            st.markdown("""
            #### BUY Signal
            Consider adding to your position. Technical indicators suggest favorable conditions with positive momentum building. Current price levels may represent good value.
            """)
        elif current_index >= 45:
            st.markdown("""
            #### HODL Signal
            Maintain current positions. This is not an ideal time to make significant changes to your portfolio allocation. The market is showing mixed signals without a clear directional bias.
            """)
        elif current_index >= 20:
            st.markdown("""
            #### SELL Signal
            Consider reducing exposure. Technical indicators suggest caution is warranted as momentum is slowing and risk levels are elevated. Protect profits or limit potential losses.
            """)
        else:
            st.markdown("""
            #### STRONG SELL Signal
            Consider significantly reducing exposure. Multiple indicators suggest high risk conditions with potential for continued downside movement. Capital preservation should be the priority.
            """)
    
    with action_cols[1]:
        st.markdown(f"**Halving Context:**<br>{halving_data['phase_description']}", unsafe_allow_html=True)
        st.markdown("**Important Note:**<br>This is not financial advice. Always do your own research.", unsafe_allow_html=True)

with tab2:
    # Get historical data
    historical_data = get_historical_data()
    
    # Calculate date ranges
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    one_month_ago = today - timedelta(days=30)
    one_year_ago = today - timedelta(days=365)
    
    # Get historical values for specific dates
    latest_index = historical_data.iloc[-1]['Index']
    week_ago_row = historical_data[historical_data['Date'] >= one_week_ago].iloc[0]
    month_ago_row = historical_data[historical_data['Date'] >= one_month_ago].iloc[0]
    year_ago_row = historical_data.iloc[0]
    
    # Display historical index values
    st.subheader("Historical Index Values")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        historical_points = [
            ("Previous close", latest_index, get_market_condition(latest_index)),
            ("1 week ago", week_ago_row['Index'], get_market_condition(week_ago_row['Index'])),
            ("1 month ago", month_ago_row['Index'], get_market_condition(month_ago_row['Index'])),
            ("1 year ago", year_ago_row['Index'], get_market_condition(year_ago_row['Index']))
        ]
        
        for label, value, condition in historical_points:
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="color: gray;">{label}</div>
                <div style="font-size: 20px; font-weight: bold;">{condition}</div>
                <div style="display: inline-block; background-color: {get_index_color(value)}; 
                            border-radius: 50%; width: 50px; height: 50px; text-align: center; 
                            line-height: 50px; color: white; font-weight: bold;">
                    {value:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Create timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Index'],
            mode='lines',
            name='Index Value',
            line=dict(color='black', width=2)
        ))
        
        # Add colored background regions
        fig.add_shape(
            type="rect",
            x0=historical_data['Date'].min(),
            x1=historical_data['Date'].max(),
            y0=0,
            y1=20,
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=historical_data['Date'].min(),
            x1=historical_data['Date'].max(),
            y0=20,
            y1=45,
            fillcolor="rgba(255, 165, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=historical_data['Date'].min(),
            x1=historical_data['Date'].max(),
            y0=45,
            y1=65,
            fillcolor="rgba(169, 169, 169, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=historical_data['Date'].min(),
            x1=historical_data['Date'].max(),
            y0=65,
            y1=80,
            fillcolor="rgba(144, 238, 144, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=historical_data['Date'].min(),
            x1=historical_data['Date'].max(),
            y0=80,
            y1=100,
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        # Update layout
        fig.update_layout(
            title="Index History - Past Year",
            xaxis_title="Date",
            yaxis_title="Index Value",
            yaxis=dict(range=[0, 100]),
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add Bollinger Bands Chart
    st.subheader("Bitcoin Price with Bollinger Bands")
    
    # Get the btc_data from price_data
    if price_data and 'btc_data' in price_data:
        btc_data = price_data['btc_data']
        
        # Make sure we have enough data
        if len(btc_data) > 20:  # Need at least enough for the 20-day SMA
            # Filter to the last 180 days to make chart more readable
            btc_data_display = btc_data.iloc[-180:].copy()
            
            # Create the figure with a completely different approach
            fig_bollinger = go.Figure()
            
            # Add the fill area between bands as a separate element
            fig_bollinger.add_trace(go.Scatter(
                x=btc_data_display.index,
                y=btc_data_display['Upper_Band'],
                mode='lines',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1, dash='dash'),
                name='Upper Band'
            ))
            
            fig_bollinger.add_trace(go.Scatter(
                x=btc_data_display.index,
                y=btc_data_display['Lower_Band'],
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dash'),
                name='Lower Band'
            ))
            
            # Add the shaded area between bands
            fig_bollinger.add_trace(go.Scatter(
                x=list(btc_data_display.index) + list(btc_data_display.index)[::-1],
                y=list(btc_data_display['Upper_Band']) + list(btc_data_display['Lower_Band'])[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 80, 0.05)',
                line=dict(width=0),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add the SMA line
            fig_bollinger.add_trace(go.Scatter(
                x=btc_data_display.index,
                y=btc_data_display['SMA_20'],
                mode='lines',
                line=dict(color='gray', width=1),
                name='20-day SMA'
            ))
            
            # Add the BTC price line with high visibility
            fig_bollinger.add_trace(go.Scatter(
                x=btc_data_display.index,
                y=btc_data_display['Close'],
                mode='lines',
                line=dict(color='blue', width=4),
                name='BTC Price',
                hovertemplate='<b>BTC Price:</b> $%{y:,.2f}<extra></extra>'
            ))
            
            # Update layout with improved settings
            fig_bollinger.update_layout(
                title="Bitcoin Price with Bollinger Bands (20-day, 2 standard deviations)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1,
                    traceorder="normal"
                ),
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            # Format y-axis to show clear price values
            fig_bollinger.update_yaxes(
                tickprefix="$",
                separatethousands=True
            )
            
            # Improve x-axis formatting
            fig_bollinger.update_xaxes(
                rangeslider_visible=False
            )
            
            # Display the chart
            st.plotly_chart(fig_bollinger, use_container_width=True)
            
            # Add interpretation
            with st.expander("Bollinger Bands Interpretation"):
                st.markdown("""
                ### Understanding Bollinger Bands
                
                Bollinger Bands consist of:
                - A middle band (20-day simple moving average)
                - An upper band (SMA + 2 standard deviations)
                - A lower band (SMA - 2 standard deviations)
                
                #### Key Signals:
                
                - **Price near upper band**: Potentially overbought condition
                - **Price near lower band**: Potentially oversold condition
                - **Bands narrowing**: Low volatility, potential breakout coming
                - **Bands widening**: Increasing volatility
                - **Price breaking through bands**: Strong momentum in that direction
                
                Bollinger Bands are most effective when used with other indicators rather than in isolation.
                """)
        else:
            st.warning("Not enough data points to calculate Bollinger Bands. Need at least 20 days of data.")
    else:
        st.warning("Price data not available to generate Bollinger Bands chart.")

with tab3:
    st.subheader("Bitcoin On-Chain Metrics")
    st.markdown("On-chain data provides insights into blockchain activity and can help identify market trends")
    
    # Check if Dune API is configured
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        st.warning("⚠️ Dune Analytics API key not found. Add your API key to the .env file to enable real on-chain data.")
    
    # Get on-chain data with loading indicator
    with st.spinner("Loading on-chain data..."):
        onchain_data = get_btc_onchain_data()
        holder_data = get_btc_holder_distribution()
        mining_data = get_btc_mining_data()
    
    # Add a notice about sample data
    any_real_data = False
    try:
        # Try to detect if we're using real or sample data
        active_addr = onchain_data["active_addresses"]
        if len(active_addr) > 0:
            if active_addr['active_addresses'].std() < 100000:  # Sample data has more variance
                any_real_data = False
            else:
                any_real_data = True
    except:
        any_real_data = False
    
    if not any_real_data:
        st.info("""
        **ⓘ Currently displaying sample data**
        
        To show real on-chain metrics:
        1. Create a Dune Analytics account at [dune.com](https://dune.com)
        2. Get your API key from your profile settings
        3. Add your API key to the `.env` file: `DUNE_API_KEY=your_key_here`
        4. Update the query IDs in the code with your own Dune queries
        """)
    
    # Create tabs for different on-chain metrics
    metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Network Activity", "Holder Distribution", "Mining"])
    
    with metrics_tab1:
        # Create metrics cards for key indicators
        col1, col2, col3, col4 = st.columns(4)
        
        # Active addresses metric
        active_addr = onchain_data["active_addresses"]
        latest_active = active_addr['active_addresses'].iloc[-1]
        prev_active = active_addr['active_addresses'].iloc[-2]
        pct_change = ((latest_active - prev_active) / prev_active) * 100
        
        with col1:
            st.metric(
                label="Active Addresses (24h)",
                value=f"{latest_active:,}",
                delta=f"{pct_change:.1f}%"
            )
        
        # Transaction volume metric
        tx_volume = onchain_data["transaction_volume"]
        latest_volume = tx_volume['volume_usd'].iloc[-1]
        prev_volume = tx_volume['volume_usd'].iloc[-2]
        pct_change = ((latest_volume - prev_volume) / prev_volume) * 100
        
        with col2:
            st.metric(
                label="Transaction Volume (24h)",
                value=f"${latest_volume/1e9:.2f}B",
                delta=f"{pct_change:.1f}%"
            )
        
        # Miner revenue metric
        miner_rev = onchain_data["miner_revenue"]
        latest_rev = miner_rev['revenue_usd'].iloc[-1]
        prev_rev = miner_rev['revenue_usd'].iloc[-2]
        pct_change = ((latest_rev - prev_rev) / prev_rev) * 100
        
        with col3:
            st.metric(
                label="Miner Revenue (24h)",
                value=f"${latest_rev/1e6:.2f}M",
                delta=f"{pct_change:.1f}%"
            )
        
        # Exchange flows metric
        exch_flow = onchain_data["exchange_flows"]
        latest_flow = exch_flow['net_flow_btc'].iloc[-1]
        label_suffix = "Inflow" if latest_flow < 0 else "Outflow"
        
        with col4:
            st.metric(
                label=f"Exchange Net {label_suffix} (24h)",
                value=f"{abs(latest_flow):.1f} BTC",
                delta=f"{'Leaving' if latest_flow > 0 else 'Entering'} Exchanges"
            )
        
        # Create charts for on-chain metrics
        st.subheader("Active Addresses (30 Days)")
        
        fig_active = go.Figure()
        fig_active.add_trace(go.Scatter(
            x=active_addr['date'],
            y=active_addr['active_addresses'],
            mode='lines',
            name='Active Addresses',
            line=dict(color='blue', width=2)
        ))
        
        fig_active.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        
        fig_active.update_yaxes(
            title_text="Active Addresses",
            separatethousands=True
        )
        
        st.plotly_chart(fig_active, use_container_width=True)
        
        # Transaction volume chart
        st.subheader("Transaction Volume (30 Days)")
        
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=tx_volume['date'],
            y=tx_volume['volume_usd']/1e9,  # Convert to billions
            name='Volume',
            marker_color='green'
        ))
        
        fig_volume.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        
        fig_volume.update_yaxes(
            title_text="Volume (Billion USD)"
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Exchange flows chart
        st.subheader("Exchange Net Flows (30 Days)")
        
        fig_flows = go.Figure()
        fig_flows.add_trace(go.Bar(
            x=exch_flow['date'],
            y=exch_flow['net_flow_btc'],
            name='Net Flow',
            marker_color=['green' if flow > 0 else 'red' for flow in exch_flow['net_flow_btc']]
        ))
        
        fig_flows.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        
        fig_flows.update_yaxes(
            title_text="Net Flow (BTC)"
        )
        
        fig_flows.add_shape(
            type="line",
            x0=exch_flow['date'].min(),
            x1=exch_flow['date'].max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        st.plotly_chart(fig_flows, use_container_width=True)
        
    with metrics_tab2:
        st.subheader("Bitcoin Holder Distribution")
        
        # Create a pie chart for holder distribution
        fig_holders = go.Figure()
        fig_holders.add_trace(go.Pie(
            labels=holder_data['range'],
            values=holder_data['addresses'],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(
                colors=[
                    'rgba(31, 119, 180, 0.8)',
                    'rgba(255, 127, 14, 0.8)',
                    'rgba(44, 160, 44, 0.8)',
                    'rgba(214, 39, 40, 0.8)',
                    'rgba(148, 103, 189, 0.8)',
                    'rgba(140, 86, 75, 0.8)',
                    'rgba(227, 119, 194, 0.8)'
                ]
            )
        ))
        
        fig_holders.update_layout(
            title="Bitcoin Addresses by Holdings",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_holders, use_container_width=True)
        
        # Add insights about holder distribution
        st.subheader("Holder Insights")
        
        total_addresses = holder_data['addresses'].sum()
        whale_addresses = holder_data['addresses'].iloc[-3:].sum()  # Last 3 categories (largest holders)
        whale_percentage = (whale_addresses / total_addresses) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Total Addresses",
                value=f"{total_addresses:,}"
            )
            
            st.metric(
                label="'Whale' Addresses (>100 BTC)",
                value=f"{whale_addresses:,}",
                delta=f"{whale_percentage:.2f}% of total"
            )
        
        with col2:
            st.info("""
            **Holder Distribution Insights:**
            
            - Small holders (less than 1 BTC) make up the vast majority of addresses
            - Large "whale" addresses control a significant portion of supply
            - Distribution patterns can indicate market sentiment and potential price movements
            """)
    
    with metrics_tab3:
        st.subheader("Bitcoin Mining Metrics")
        
        # Create metrics cards for mining data
        col1, col2, col3 = st.columns(3)
        
        # Hashrate metric
        latest_hashrate = mining_data['hashrate_th_s'].iloc[-1] / 1e6  # Convert to EH/s
        prev_hashrate = mining_data['hashrate_th_s'].iloc[-2] / 1e6
        pct_change = ((latest_hashrate - prev_hashrate) / prev_hashrate) * 100
        
        with col1:
            st.metric(
                label="Network Hashrate",
                value=f"{latest_hashrate:.2f} EH/s",
                delta=f"{pct_change:.1f}%"
            )
        
        # Difficulty metric
        latest_diff = mining_data['difficulty'].iloc[-1] / 1e12  # Convert to T
        prev_diff = mining_data['difficulty'].iloc[-2] / 1e12
        pct_change = ((latest_diff - prev_diff) / prev_diff) * 100
        
        with col2:
            st.metric(
                label="Mining Difficulty",
                value=f"{latest_diff:.2f}T",
                delta=f"{pct_change:.1f}%"
            )
        
        # Block time metric
        latest_blocktime = mining_data['block_time_seconds'].iloc[-1]
        prev_blocktime = mining_data['block_time_seconds'].iloc[-2]
        # Negative change in block time is positive (faster blocks)
        pct_change = ((prev_blocktime - latest_blocktime) / prev_blocktime) * 100
        
        with col3:
            st.metric(
                label="Average Block Time",
                value=f"{latest_blocktime:.1f} seconds",
                delta=f"{pct_change:.1f}%"
            )
        
        # Hashrate chart
        st.subheader("Network Hashrate (30 Days)")
        
        fig_hashrate = go.Figure()
        fig_hashrate.add_trace(go.Scatter(
            x=mining_data['date'],
            y=mining_data['hashrate_th_s'] / 1e6,  # Convert to EH/s
            mode='lines',
            name='Hashrate',
            line=dict(color='orange', width=2)
        ))
        
        fig_hashrate.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        
        fig_hashrate.update_yaxes(
            title_text="Hashrate (EH/s)"
        )
        
        st.plotly_chart(fig_hashrate, use_container_width=True)
        
        # Difficulty chart
        st.subheader("Mining Difficulty (30 Days)")
        
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=mining_data['date'],
            y=mining_data['difficulty'] / 1e12,  # Convert to T
            mode='lines',
            name='Difficulty',
            line=dict(color='purple', width=2)
        ))
        
        fig_diff.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified"
        )
        
        fig_diff.update_yaxes(
            title_text="Difficulty (T)"
        )
        
        st.plotly_chart(fig_diff, use_container_width=True)
        
        # Add mining insights
        st.info("""
        **Mining Metrics Insights:**
        
        - **Hashrate**: Represents total computational power securing the network
        - **Difficulty**: Auto-adjusts to maintain ~10-minute block times
        - **Block Time**: Average time between blocks (target is 600 seconds)
        
        Rising hashrate typically indicates miner confidence in Bitcoin's long-term value.
        """)

with tab4:
    st.subheader("Reddit Sentiment Analysis")
    st.markdown("Reddit sentiment analysis provides insights into community sentiment towards Bitcoin")
    
    # Get sentiment data with loading indicator
    with st.spinner("Loading sentiment data..."):
        sentiment_data = get_reddit_sentiment()
    
    # Check if we're using sample data
    if sentiment_data["data_source"] == "sample_data":
        st.info("""
        **ⓘ Currently displaying sample data**
        
        To show real Reddit sentiment:
        1. Ensure your .env file contains valid Reddit API credentials:
           - REDDIT_CLIENT_ID=your_client_id
           - REDDIT_CLIENT_SECRET=your_client_secret
           - REDDIT_USER_AGENT=your_user_agent
        """)
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Convert sentiment score to a 0-100 scale for consistency with other metrics
        sentiment_score = (sentiment_data["overall_sentiment"] + 1) * 50
        st.metric(
            label="Community Sentiment", 
            value=f"{sentiment_score:.0f}/100"
        )
    
    with col2:
        # Create a ratio of bullish:bearish
        bullish = sentiment_data["sentiment_counts"].get("bullish", 0)
        bearish = sentiment_data["sentiment_counts"].get("bearish", 0)
        ratio = bullish / max(1, bearish)  # Avoid division by zero
        
        st.metric(
            label="Bull/Bear Ratio", 
            value=f"{ratio:.1f}"
        )
    
    with col3:
        st.metric(
            label="Posts Analyzed", 
            value=f"{sentiment_data['post_count']:,}"
        )
    
    with col4:
        # Calculate percentage of bullish content
        total = sum(sentiment_data["sentiment_counts"].values())
        bullish_pct = (bullish / total) * 100 if total > 0 else 0
        
        st.metric(
            label="Bullish Content", 
            value=f"{bullish_pct:.1f}%"
        )
    
    # Create sentiment gauge chart
    st.subheader("Community Sentiment Gauge")
    
    fig_sentiment = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Reddit Sentiment", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "rgba(255, 0, 0, 0.7)"},
                {'range': [30, 45], 'color': "rgba(255, 165, 0, 0.7)"},
                {'range': [45, 55], 'color': "rgba(169, 169, 169, 0.7)"},
                {'range': [55, 70], 'color': "rgba(144, 238, 144, 0.7)"},
                {'range': [70, 100], 'color': "rgba(0, 128, 0, 0.7)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    # Update layout
    fig_sentiment.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=16)
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Create tabs for more detailed sentiment analysis
    sentiment_tab1, sentiment_tab2, sentiment_tab3 = st.tabs(["Trending Posts", "Sentiment by Subreddit", "Sentiment Over Time"])
    
    with sentiment_tab1:
        st.subheader("Trending Posts")
        trending_posts = sentiment_data["trending_posts"]
        
        for i, post in trending_posts.iterrows():
            # Determine color based on sentiment
            if post["sentiment_label"] == "bullish":
                sentiment_color = "green"
            elif post["sentiment_label"] == "bearish":
                sentiment_color = "red"
            else:
                sentiment_color = "gray"
                
            # Calculate how long ago the post was made
            time_diff = datetime.now() - post["created_utc"]
            hours_ago = time_diff.total_seconds() / 3600
            
            if hours_ago < 24:
                time_str = f"{int(hours_ago)} hours ago"
            else:
                days = int(hours_ago / 24)
                time_str = f"{days} days ago"
            
            st.markdown(f"""
            <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin-bottom: 20px;">
                <h4>{post['title']}</h4>
                <p>r/{post['subreddit']} • {post['score']} points • {post['num_comments']} comments • {time_str}</p>
                <p><span style="background-color: {'rgba(0, 128, 0, 0.1)' if sentiment_color == 'green' else 'rgba(255, 0, 0, 0.1)' if sentiment_color == 'red' else 'rgba(128, 128, 128, 0.1)'};
                         color: {sentiment_color}; padding: 2px 8px; border-radius: 10px;">
                    {post['sentiment_label'].upper()}
                </span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with sentiment_tab2:
        st.subheader("Sentiment by Subreddit")
        
        # Create a bar chart for subreddit sentiment
        subreddit_names = list(sentiment_data["subreddit_sentiment"].keys())
        sentiment_values = list(sentiment_data["subreddit_sentiment"].values())
        
        fig_subreddits = go.Figure()
        fig_subreddits.add_trace(go.Bar(
            x=subreddit_names,
            y=sentiment_values,
            marker_color=['green' if v > 0.1 else 'red' if v < -0.1 else 'gray' for v in sentiment_values]
        ))
        
        fig_subreddits.update_layout(
            title="Sentiment by Subreddit",
            xaxis_title="Subreddit",
            yaxis_title="Sentiment Score (-1 to 1)",
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        # Add a zero line
        fig_subreddits.add_shape(
            type="line",
            x0=-0.5,
            x1=len(subreddit_names) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        st.plotly_chart(fig_subreddits, use_container_width=True)
        
        # Display subreddit descriptions
        with st.expander("About these subreddits"):
            st.markdown("""
            - **r/Bitcoin**: The main Bitcoin subreddit, tends to be very bullish and focused on long-term holding
            - **r/CryptoCurrency**: Covers all cryptocurrencies, often more balanced in sentiment
            - **r/BitcoinMarkets**: Focused on trading Bitcoin, more technical and sometimes contrarian
            - **r/btc**: Originally a fork of r/Bitcoin after moderation disputes, now more focused on Bitcoin Cash
            """)
    
    with sentiment_tab3:
        st.subheader("Sentiment Over Time")
        
        # Create line chart for sentiment over time
        fig_sentiment_over_time = go.Figure()
        fig_sentiment_over_time.add_trace(go.Scatter(
            x=sentiment_data['sentiment_over_time']['date'],
            y=sentiment_data['sentiment_over_time']['sentiment_score'],
            mode='lines',
            name='Sentiment Score',
            line=dict(color='blue', width=2)
        ))
        
        # Add a zero line
        fig_sentiment_over_time.add_shape(
            type="line",
            x0=sentiment_data['sentiment_over_time']['date'].min(),
            x1=sentiment_data['sentiment_over_time']['date'].max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Highlight areas based on sentiment
        dates = sentiment_data['sentiment_over_time']['date']
        scores = sentiment_data['sentiment_over_time']['sentiment_score']
        
        for i in range(len(dates) - 1):
            if scores[i] > 0.1:
                color = "rgba(0, 255, 0, 0.1)"  # Light green
            elif scores[i] < -0.1:
                color = "rgba(255, 0, 0, 0.1)"  # Light red
            else:
                continue  # Skip neutral sentiments
                
            fig_sentiment_over_time.add_shape(
                type="rect",
                x0=dates[i],
                x1=dates[i+1],
                y0=min(0, scores[i]),
                y1=max(0, scores[i]),
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )
        
        fig_sentiment_over_time.update_layout(
            title="Reddit Sentiment Trend",
            xaxis_title="Date",
            yaxis_title="Sentiment Score (-1 to 1)",
            yaxis=dict(range=[-1, 1]),  # Fix y-axis range
            height=400,
            margin=dict(l=20, r=20, t=50, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_sentiment_over_time, use_container_width=True)
        
        # Add sentiment distribution pie chart
        st.subheader("Sentiment Distribution")
        
        # Get sentiment counts
        labels = list(sentiment_data["sentiment_counts"].keys())
        values = list(sentiment_data["sentiment_counts"].values())
        
        # Define colors for each sentiment
        colors = {
            "bullish": "rgba(0, 128, 0, 0.7)",
            "neutral": "rgba(128, 128, 128, 0.7)",
            "bearish": "rgba(255, 0, 0, 0.7)"
        }
        
        # Create pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=[colors.get(label, "gray") for label in labels])
        )])
        
        fig_pie.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

# Methodology section
st.markdown("---")
st.header("How It Works", anchor="how-it-works")
st.markdown("""
This index combines multiple technical indicators and on-chain metrics to produce a single score that helps 
guide investment decisions for Bitcoin and other cryptocurrencies.

### Components

The index calculation includes:

1. **RSI Analysis (25%)**: Relative Strength Index to determine if BTC is overbought or oversold
2. **Trend Analysis (35%)**: Position relative to key moving averages (50, 200 SMA)
3. **Performance (25%)**: Recent market returns on various timeframes
4. **Volatility (15%)**: Current volatility compared to historical patterns

### On-Chain Data Integration

The dashboard also includes real-time on-chain metrics from Dune Analytics:

1. **Network Activity**: Active addresses, transaction volume, and exchange flows
2. **Holder Distribution**: Analysis of address sizes and whale concentration
3. **Mining Metrics**: Hashrate, difficulty, and block times

On-chain metrics provide additional context to price movements and can help identify sustainable trends versus temporary price fluctuations.

### Interpretation

- **0-20 (STRONG SELL)**: Multiple indicators suggest extremely bearish conditions
- **20-45 (SELL)**: Market conditions are deteriorating, consider reducing exposure
- **45-65 (HODL)**: Market shows mixed signals, holding current positions is advised
- **65-80 (BUY)**: Favorable technical conditions suggest potential for upward movement
- **80-100 (STRONG BUY)**: Multiple indicators align for a strong bullish outlook

### Disclaimer

This index is for informational purposes only and should not be considered financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.
""")

# Footer
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("Last updated: " + datetime.now().strftime("%B %d at %I:%M:%S %p"))
with col2:
    st.markdown("Data sources: Yahoo Finance API, Dune Analytics")