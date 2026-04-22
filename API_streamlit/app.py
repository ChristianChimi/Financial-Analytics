import streamlit as st
import plotly.graph_objects as go
from utils import get_robust_data, create_future_exog
from model import RobustForecastModel

st.set_page_config(page_title="AI Quant Pro", layout="wide", page_icon="📉")

st.title("🛡️ Advanced Financial Forecasting")
st.markdown("Analisi neurale multivariata con orizzonte grafico esteso a 2 anni.")

st.sidebar.header("Impostazioni Analisi")
ticker = st.sidebar.text_input("Ticker Simbolo", "AAPL").upper()
period = st.sidebar.selectbox("Dati Storici (Lookback)", ["1y", "2y", "3y", "5y", "10y"], index=2)
horizon = st.sidebar.slider("Orizzonte Predizione (Giorni)", 7, 90, 30)

@st.cache_resource
def load_model_instance(h):
    return RobustForecastModel(horizon=h)

if st.sidebar.button("Esegui Analisi"):
    with st.spinner(f"Elaborazione dati per {ticker}..."):
        data = get_robust_data(ticker, period=period)
        
        if data is not None:
            futr_df = create_future_exog(data, horizon)
            model_engine = load_model_instance(horizon)
            forecast = model_engine.train_and_predict(data, futr_df)
            
            try:
                col_p10 = [c for c in forecast.columns if '0.1' in c or 'lo' in c][0]
                col_p50 = [c for c in forecast.columns if '0.5' in c or 'median' in c or 'NHITS' == c][0]
                col_p90 = [c for c in forecast.columns if '0.9' in c or 'hi' in c][0]
            except IndexError:
                st.error("Errore nella mappatura delle colonne del modello.")
                st.stop()

            fig = go.Figure()
            
            # --- MODIFICA: Visualizzazione di 2 anni di storico (circa 500 giorni di borsa) ---
            hist = data.tail(504) 
            
            fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], 
                                     name="Storico (2 anni)", line=dict(color='#636efa', width=1.5)))
            
            # Area di Incertezza
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast[col_p90].tolist() + forecast[col_p10].tolist()[::-1],
                fill='toself', fillcolor='rgba(0, 255, 255, 0.1)', 
                line=dict(color='rgba(255,255,255,0)'), name="Incertezza (80%)"
            ))
            
            # Predizione
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[col_p50], 
                                     name="Predizione", line=dict(color='cyan', width=3)))
            
            fig.update_layout(template="plotly_dark", hovermode="x unified", height=600,
                              xaxis=dict(rangeselector=dict(buttons=list([
                                  dict(count=6, label="6m", step="month", stepmode="backward"),
                                  dict(count=1, label="1y", step="year", stepmode="backward"),
                                  dict(step="all", label="2y")
                              ])), type="date"))
            
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            ultimo_prezzo = data['y'].iloc[-1]
            target_finale = forecast[col_p50].iloc[-1]
            c1.metric("Prezzo Attuale", f"${ultimo_prezzo:.2f}")
            c2.metric("Target Previsto", f"${target_finale:.2f}")
            c3.metric("Rendimento Potenziale", f"{((target_finale/ultimo_prezzo)-1)*100:.2f}%")
            
        else:
            st.error("Impossibile recuperare i dati.")