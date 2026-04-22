import torch
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import QuantileLoss

class RobustForecastModel:
    def __init__(self, horizon=30):
        self.horizon = horizon
        self.futr_exog = ['sp500', 'vix', 'volume', 'vol_ma']
        quantiles_tensor = torch.tensor([0.1, 0.5, 0.9])
        
        self.model = NHITS(
            h=self.horizon,
            input_size=60,         
            futr_exog_list=self.futr_exog,
            scaler_type='robust',
            max_steps=50,          
            accelerator='cpu',     
            loss=QuantileLoss(q=quantiles_tensor)
        )
        self.nf = NeuralForecast(models=[self.model], freq='B')

    def _prepare_and_clean_future(self, train_df, reference_df):
        """Genera futr_df e garantisce l'assenza totale di valori nulli"""
        # 1. Genera le date attese dal modello
        futr_df = self.nf.make_future_dataframe(df=train_df)
        
        # 2. Merge con i dati di riferimento (reference_df contiene i valori esogeni)
        # Usiamo solo le colonne che ci servono per evitare conflitti
        cols_to_merge = ['ds'] + [c for c in self.futr_exog if c in reference_df.columns]
        futr_df = pd.merge(futr_df[['unique_id', 'ds']], 
                           reference_df[cols_to_merge], 
                           on='ds', how='left')
        
        # 3. PULIZIA AGGRESSIVA (Triple Pass)
        # Passaggio 1: Trascina l'ultimo valore noto (Forward Fill)
        # Passaggio 2: Se mancano i primi valori, usa i successivi (Backward Fill)
        # Passaggio 3: Se ancora nullo (es. esogene vuote), usa l'ultimo valore del training
        futr_df = futr_df.ffill().bfill()
        
        for col in self.futr_exog:
            if futr_df[col].isnull().any():
                last_known = train_df[col].iloc[-1]
                futr_df[col] = futr_df[col].fillna(last_known)
            # Protezione finale estrema: se è ancora NaN, metti 0
            futr_df[col] = futr_df[col].fillna(0)
            
        return futr_df

    def train_and_predict(self, df, _unused):
        # Pulizia dati di training
        df_clean = df.copy().ffill().bfill().fillna(0)
        self.nf.fit(df=df_clean)
        
        # Creazione e pulizia futuro
        correct_futr = self._prepare_and_clean_future(df_clean, df_clean)
        
        forecasts = self.nf.predict(futr_df=correct_futr)
        return forecasts.reset_index()

    def run_backtest(self, df):
        # Pulizia dati base
        df_clean = df.copy().ffill().bfill().fillna(0)
        
        # Split per backtest
        train_df = df_clean.iloc[:-30].copy()
        test_df = df_clean.iloc[-30:].copy()
        
        self.nf.fit(df=train_df)
        
        # Per il backtest usiamo test_df come riferimento per le esogene reali
        correct_futr_bt = self._prepare_and_clean_future(train_df, df_clean)
        
        backtest_forecast = self.nf.predict(futr_df=correct_futr_bt)
        return backtest_forecast.reset_index(), test_df