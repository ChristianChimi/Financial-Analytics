from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.losses import Huber

def build_lstm_model(input_shape):
    """Initializes a Stacked LSTM model with LeakyReLU activation and Huber Loss."""
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        LSTM(units=64, return_sequences=False),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(units=32),
        LeakyReLU(alpha=0.1),
        Dense(units=1)
    ])
    
    # Robust loss function for financial time series
    model.compile(optimizer='adam', loss=Huber(delta=1.0))
    return model