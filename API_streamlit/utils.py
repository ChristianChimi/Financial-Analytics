import yfinance as yf
import pandas as pd
import streamlit as st

def get_robust_data(ticker, period="3y"):
    try:
        stock_raw = yf.download(ticker, period=period, auto_adjust=True)
        if stock_raw.empty: return None
        
        df = stock_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'volume'})
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

        sp500_raw = yf.download('^GSPC', period=period, auto_adjust=True)
        vix_raw = yf.download('^VIX', period=period, auto_adjust=True)

        for temp in [sp500_raw, vix_raw]:
            if isinstance(temp.columns, pd.MultiIndex): temp.columns = temp.columns.get_level_values(0)

        sp500 = sp500_raw[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'sp500'})
        vix = vix_raw[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'vix'})
        
        for temp in [sp500, vix]: temp['ds'] = pd.to_datetime(temp['ds']).dt.tz_localize(None)

        final_df = pd.merge(df, sp500, on='ds', how='left')
        final_df = pd.merge(final_df, vix, on='ds', how='left')
        final_df['vol_ma'] = final_df['volume'].rolling(window=5).mean()
        final_df = final_df.ffill().bfill().fillna(0)
        final_df['unique_id'] = ticker
        
        return final_df[['unique_id', 'ds', 'y', 'volume', 'sp500', 'vix', 'vol_ma']]
    except Exception as e:
        st.error(f"Errore: {e}")
        return None

def create_future_exog(df, horizon):
    last_row = df.iloc[-1]
    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    return pd.DataFrame({
        'ds': future_dates,
        'unique_id': df['unique_id'].iloc[0],
        'sp500': last_row['sp500'], 'vix': last_row['vix'],
        'volume': last_row['volume'], 'vol_ma': last_row['vol_ma']
    })