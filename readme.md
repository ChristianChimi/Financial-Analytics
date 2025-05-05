## **Financial Analytics - Stock Market**
## **Overview**
In this project, I analyze 10 years of historical data for various financial instruments, including NASDAQ, S&P 500, iShares MSCI World (EIMI), Apple stock, and 5 years of VWCE data. The goal was to gain insights into stock market behavior and portfolio performance.

Real datasets from investing.com!

## **Exploratory Data Analysis**
- Correlation and scatterplot between NASDAQ, S&P500 and MSCI World.
- Normalization of prices to efficiently plot NASDAQ, MSCI World and S&P500 on the same graph.
- Comparison Indexes-Apple stock.
- Analyzed the correlation between MSCI World and EIMI indexes. Are these two indices truly providing diversification?

## **Deep Learning Models**
 - I trained two different neural networks using PyTorch to assess predictive capabilities:
    - Price prediction with torch:
      - MLP: predict Apple stock price based on S&P 500 and NASDAQ prices.
       Also available as API here! https://huggingface.co/spaces/ChristianChimi/Apple-Stock-API
    - LSTM: Used an LSTM model to predict the future price of the MSCI World Index using time series data.
      - Model Evaluation: Monitored and printed the loss and the validation loss function during training to avoid overfitting.
        - Plotted loss and validation loss over the epochs.
        - Hyperparameters tuning: hidden layers, hidden layers size, learning rate, epochs, dropout.
          
## **Portfolio simulation** 
   - Lump sum investment and dollar cost averaging with performances.
   - Financial performance metrics: Performance evaluated using:
   - CAGR (Compound Annual Growth Rate)
   - ROI (Return on Investment)
   - Sharpe Ratio (Risk-adjusted return)
     
## **Key Insights** 
  - Nasdaq-100, MSCI World and S&P500 have a nearly perfect correlation. Nasdaq and S&P have slightly better performance but higher volatility.
  - Normalizing S&P500 and Apple stock prices, we clearly see Apple tracks the index most of the time.
  - Emerging markets (EIMI) do not currently offer effective diversification — in the last year, they’ve shown high correlation with developed markets, along with lower returns and higher volatility.

## **Technologies Used**
 - **Python**, **Pandas**, **Numpy**, **PyTorch**, **Scikit-learn**

## **Conclusion**
This project reveals the high interconnectedness of global markets and the limitations of traditional diversification strategies. While deep learning models can identify general trends in stock behavior, their predictive power remains constrained by market complexity. Portfolio simulations confirm that investment strategies must be tailored to both market conditions and individual risk tolerance.
