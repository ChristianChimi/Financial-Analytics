## **Financial Analytics - Stock Market**
## **Overviwew**
In this project, I analyze 10 years of historical data for various financial instruments, including NASDAQ, S&P 500, iShares MSCI World (EIMI), Apple stock, and 5 years of VWCE data. The goal was to gain insights into stock market behavior and portfolio performance.

## **Exploratory Data Analysis**
     - Correlation and scatterplot between NASDAQ, S&P500 and MSCI World.
     - Normalization of prices to efficiently plot NASDAQ, MSCI World and S&P500 on the same graph.
     - Comparison Indexes-Apple stock.
     - Volatility evaluation: Evaluated the volatility of NASDAQ, S&P 500, MSCI World, and Apple stock
     - Analyzed the correlation between MSCI World and EIMI indexes. Are these two indices truly providing diversification?

## **Deep Learning**
     - Price prediction with torch:
          - MLP: predict APPLE stock price basing on S&P500 and NASDAQ prices.
            Also available as API here! https://huggingface.co/spaces/ChristianChimi/Apple-Stock-API
          - LSTM: Used an LSTM model to predict the future price of the MSCI World Index using time series data.
          - Model Evaluation: Monitored and printed the loss function during training.
          
## **Portfolio simulation** 
     - Lump sum investment and dollar cost averaging with performances.
     - Financial performance metrics: CAGR, ROI and sharpe ratio.
   
## **Technologies Used**
     - **Python**, **Pandas**, **Numpy**, **PyTorch**, **Scikit-learn**
