import torch
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

    def train_and_predict(self, df, futr_df):
        df_clean = df.copy().fillna(0)
        futr_clean = futr_df.copy().fillna(0)
        self.nf.fit(df=df_clean)
        forecasts = self.nf.predict(futr_df=futr_clean)
        return forecasts.reset_index()