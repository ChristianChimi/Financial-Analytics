import streamlit as st
import plotly.graph_objects as go
from utils import get_robust_data, create_future_exog
from model import RobustForecastModel

st.set_page_config(page_title="AI Quant Pro", layout="wide")

# FIX: Forza lo svuotamento della cache se modifichi il codice del modello
if st.sidebar.button("Reset Cache Sistema"):
    st.cache_resource.clear()
    st.success("Cache pulita!")

st.title("🛡️ AI Financial Terminal")

ticker = st.sidebar.text_input("Ticker Simbolo", "NVDA").upper()
period = st.sidebar.selectbox("Storico", ["2y", "3y", "5y"], index=1)
horizon = st.sidebar.slider("Orizzonte Predizione (Giorni)", 7, 60, 30)

@st.cache_resource
def load_engine(h):
    return RobustForecastModel(horizon=h)

if st.sidebar.button("Avvia Analisi"):
    data = get_robust_data(ticker, period=period)
    
    if data is not None:
        engine = load_engine(horizon)
        
        tab1, tab2 = st.tabs(["🔮 Forecast Futuro", "📊 Backtesting (Ultimi 30gg)"])
        
        with tab1:
            with st.spinner("Generando previsione futura..."):
                futr_df = create_future_exog(data, horizon)
                forecast = engine.train_and_predict(data, futr_df)
                
                # Identificazione dinamica colonne
                col_p10 = [c for c in forecast.columns if '0.1' in c or 'lo' in c][0]
                col_p50 = [c for c in forecast.columns if '0.5' in c or 'NHITS' == c or 'median' in c][0]
                col_p90 = [c for c in forecast.columns if '0.9' in c or 'hi' in c][0]

                fig = go.Figure()
                # Storico (ultimi 2 anni nel grafico come richiesto)
                hist_plot = data.tail(504)
                fig.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['y'], name="Storico", line=dict(color='#636efa')))
                # Incertezza
                fig.add_trace(go.Scatter(x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                                         y=forecast[col_p90].tolist() + forecast[col_p10].tolist()[::-1],
                                         fill='toself', fillcolor='rgba(0, 255, 255, 0.1)', line=dict(color='rgba(0,0,0,0)'), name="Range 80%"))
                # Target
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[col_p50], name="Previsione", line=dict(color='cyan', width=3)))
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            with st.spinner("Eseguendo backtest..."):
                try:
                    bt_forecast, real_data = engine.run_backtest(data)
                    col_bt = [c for c in bt_forecast.columns if '0.5' in c or 'NHITS' == c][0]

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=real_data['ds'], y=real_data['y'], name="Prezzo Reale", line=dict(color='white')))
                    fig_bt.add_trace(go.Scatter(x=bt_forecast['ds'], y=bt_forecast[col_bt], name="AI Backtest", line=dict(color='orange', dash='dash')))
                    fig_bt.update_layout(template="plotly_dark", title="Confronto: Realtà vs Previsione AI (Mese Scorso)")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    errore = abs(real_data['y'].values - bt_forecast[col_bt].values).mean()
                    st.metric("Errore Medio (MAE)", f"${errore:.2f}")
                except Exception as e:
                    st.error(f"Errore durante il backtest: {e}")
    else:
        st.error("Ticker non trovato o errore download.")