# **Financial Analytics - Stock Market**
## **Overview**
In this project I analyzed 10 years of historical data for the most famous financial indexes,including: NASDAQ-100, S&P500, iShares MSCI World, EIMI, Apple stock, and 5 years of VWCE data.
The goal was to gain insights into stock market behavior and portfolio performance.

Real datasets from investing.com!

## **Exploratory Data Analysis**
- Correlation and scatterplot between NASDAQ-100, S&P500 and MSCI World.
- Normalization of prices to efficiently plot NASDAQ-100, MSCI World and S&P500 on the same graph.
- Comparison Indexes-Apple stock.
- Analyzed the correlation between MSCI World and EIMI indexes.

## **Deep Learning Models**
 - I trained three different neural networks using PyTorch and Neural Forecaster:
    - Robust Financial Forecasting (NHITS):
      - Developed an advanced forecasting pipeline using the NHITS architecture to predict market trends with a 30-day horizon.
      - Integrated multi-quantile loss functions (10%, 50%, 90%) to provide probabilistic uncertainty intervals and risk assessment.
      - Exogenous Feature Engineering: Optimized model accuracy by incorporating dynamic exogenous variables, including VIX, S&P 500, and volume metrics.
      - Engineered a resilient "triple-pass" data pipeline to handle financial outliers and ensure 100% data integrity through robust scaling and automated imputation.
      - Api available here https://financial-analytics-lwhmddpjchajmpnxgfv2gm.streamlit.app/
    - Price prediction with torch:
      - MLP: predict Apple stock price based on S&P 500 and NASDAQ prices.
       Also available as API here! https://huggingface.co/spaces/ChristianChimi/Apple-Stock-API
    - LSTM: Used an LSTM model to predict the future price of the MSCI World Index using time series data.
      - Model Evaluation: Monitored and printed the loss and the validation loss function during training to avoid overfitting.
        - Plotted loss and validation loss over the epochs.
        - Hyperparameters tuning: hidden layers, hidden layers size, learning rate, epochs, dropout.
          
## **Portfolio simulation** 
   - Lump sum investment and dollar cost averaging simulation.
   - Financial performance metrics, performance evaluated using:
     - CAGR (Compound Annual Growth Rate)
     - ROI (Return on Investment)
   	 - Sharpe Ratio (Risk-adjusted return)
     
## **Key Insights** 
  - Nasdaq-100, MSCI World and S&P500 have a nearly perfect correlation. Nasdaq and S&P have slightly better performance but higher volatility.
  - Normalizing S&P500 and Apple stock prices, we clearly see Apple tracks the index most of the time.
  - Emerging markets (EIMI) do not currently offer effective diversification as in the last years they’ve shown high correlation with developed markets, with lower returns and higher volatility.

## **Technologies Used**
 - **Python**, **Pandas**, **Numpy**, **PyTorch**, **Scikit-learn**, **NeuralForecaster**

## **Conclusions**
This project analized interconnection between markets and confirmed correlations between them. Performed also Deeplearning models, with acceptable loss and validation, avoiding overfitting.
