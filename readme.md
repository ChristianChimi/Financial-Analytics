Hi! This is a little project of data analysis. 
I loaded CSV containing 10 years of historical data of NASDAQ, S&P 500, iShares MSCI World. EIMI and Apple stock + 5 years of VWCE..
After doing some pre-analytics steps (Date conversion, sorting by date, deleting strings characters and conversion to float) i found/did:

- EDA:
     - Correlation and scatterplot between NASDAQ, S&P500 and MSCI World.
     - Normalization of prices to efficiently plot NASDAQ, MSCI World and S&P500 on the same graphic.
     - Comparison Indexes-Apple stock.
     - Evaluation of volatily of the 4 products.
     - Reflection on correlation between MSCI and EIMI indexes. Are we really diverisfying?

- Deep learning:
     - Price prediction with torch:
          - MLP: predict APPLE stock price basing on S&P500 and NASDAQ prices.
          - LSTM: time series to predict MSCI world index price.
          
 - Portfolio simulation: 
     - Lump sum investment and dollar coast averaging with performances.
     - Financial performance metrics: CAGR, ROI and sharpe ratio.
   
