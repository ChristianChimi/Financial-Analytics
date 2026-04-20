import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

def get_market_data(ticker):
    """
    Fetches historical market data from Yahoo Finance and validates the output.
    """
    try:
        data = yf.download(ticker, period="3y")
        
        # Safety check: if the ticker is invalid or no data is returned
        if data.empty:
            st.error(f"❌ Error: No data found for ticker '{ticker}'. Please check the symbol.")
            st.stop()
            
        # Handle MultiIndex columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data[['Close']]
    
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
        st.stop()

def prepare_dataset(data, window_size=60):
    """
    Calculates log-returns for stationarity and prepares sequences for LSTM training.
    """
    # Log-returns prevent the model from 'dropping' at all-time highs
    log_returns = np.log(data / data.shift(1)).fillna(0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_returns)
    
    X, y = [], []
    # Create sliding window sequences
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    
    return X, y, scaler, scaled_data

def recursive_forecast(model, last_window, steps, scaler, raw_data):
    """
    Generates a future trajectory using recursive inference combined 
    with historical drift and stochastic volatility.
    """
    # Calculate historical drift (mean) and volatility (std) from log-returns
    log_returns = np.log(raw_data / raw_data.shift(1)).dropna()
    avg_drift = log_returns.mean().values[0]
    hist_vol = log_returns.std().values[0]
    
    current_batch = last_window.reshape(1, 60, 1)
    log_preds = []
    
    for _ in range(steps):
        # Neural network prediction (scaled log-return)
        pred_scaled = model.predict(current_batch, verbose=0)[0]
        
        # Inject stochastic noise and drift for realistic price action
        # This prevents the 'flat line' effect in long-term forecasts
        noise = np.random.normal(avg_drift, hist_vol * 0.3)
        pred_final = pred_scaled + (noise / scaler.scale_[0])
        
        log_preds.append(pred_final)
        
        # Update the sliding window with the new prediction
        current_batch = np.append(current_batch[:, 1:, :], [[pred_final]], axis=1)
    
    # Inverse transform log-returns back to price space
    unscaled_log_returns = scaler.inverse_transform(np.array(log_preds).reshape(-1, 1))
    
    # Cumulate returns and apply to the last known price
    last_price = raw_data.values[-1][0]
    price_forecast = []
    accumulated_return = 0
    
    for r in unscaled_log_returns:
        accumulated_return += r[0]
        price_forecast.append(last_price * np.exp(accumulated_return))
    
    return np.array(price_forecast).reshape(-1, 1)