import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta
from tensorflow.keras.callbacks import EarlyStopping
from model_tft import build_lstm_model
from utils_tft import get_market_data, prepare_dataset, recursive_forecast

# UI Configuration
st.set_page_config(page_title="Neural Engine Pro", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎮 Control Panel")
ticker = st.sidebar.text_input("Ticker Symbol", "NVDA")
epochs = st.sidebar.slider("Training Epochs", 20, 100, 50)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)

st.sidebar.divider()
st.sidebar.info("Model: Stacked LSTM v3\nInput: Log-Returns\nInference: Stochastic Drift")

# --- MAIN INTERFACE ---
st.title("Neural Market Engine v3")
st.write(f"Forecasting for: **{ticker}**")

if st.button("Run Neural Training"):
    with st.status("Processing market data and training...", expanded=True) as status:
        # Data loading and prep
        raw_data = get_market_data(ticker)
        X, y, scaler, scaled_full = prepare_dataset(raw_data)
        
        # Model initialization
        status.update(label="Building LSTM Architecture...", state="running")
        model = build_lstm_model((X.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
        
        # Training loop
        p_bar = st.progress(0)
        log_area = st.empty()
        
        for e in range(epochs):
            h = model.fit(X, y, epochs=1, batch_size=32, verbose=0, callbacks=[early_stop])
            loss = h.history['loss'][0]
            p_bar.progress((e + 1) / epochs)
            log_area.code(f"Epoch {e+1}/{epochs} | Training Loss: {loss:.6f}")
            
            if model.stop_training:
                st.warning("🎯 Convergence Reached! (Early Stopping)")
                p_bar.progress(1.0)
                break
        
        status.update(label="Generating Recursive Forecasts...", state="running")
        future_preds = recursive_forecast(model, scaled_full[-60:], forecast_days, scaler, raw_data)
        status.update(label="Analysis Completed!", state="complete")

    # --- VISUALIZATION ---
    st.divider()
    last_date = raw_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    hist_tail = raw_data.tail(90)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Historical Plot
    ax.plot(hist_tail.index, hist_tail.values, color='#00d4ff', label="Historical Price", linewidth=2)
    
    # Neural Projection
    all_dates = [hist_tail.index[-1]] + future_dates
    all_vals = [hist_tail.values[-1][0]] + list(future_preds.flatten())
    ax.plot(all_dates, all_vals, color='#ff4b4b', linestyle='--', label="Neural Drift Projection", linewidth=2.5)
    
    # Styling
    ax.tick_params(colors='white')
    ax.grid(alpha=0.1, linestyle='--')
    ax.legend(facecolor='#0e1117', labelcolor='white')
    ax.set_title(f"{ticker} Forecast Analysis", color='white', fontsize=16)
    
    st.pyplot(fig)

    # Metrics Summary
    c1, c2 = st.columns(2)
    c1.metric("Current Price", f"${hist_tail.values[-1][0]:.2f}")
    delta = ((future_preds[-1][0] / hist_tail.values[-1][0]) - 1) * 100
    c2.metric("Target (End of Period)", f"${future_preds[-1][0]:.2f}", f"{delta:.2f}%")