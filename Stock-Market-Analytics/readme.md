Hi! This is a little project of data analysis. 
I loaded CSV containing 2 years of historical data of VWCE, S&P 500, iShares MSCI World  and Apple stock.
After doing some pre-analytics steps (Date conversion, sorting by date, deleting strings characters and conversion to float) i found/did:
- Indexes evaluation:
     - Correlation and scatterplot between VWCE, S&P500 and MSCI World
     - Normalization of prices to efficiently plot VWCE, MSCI World and S&P500 on the same graphic
     - Comparison Indexes-Apple stock
     - Evaluation of volatily of the 4 products
     - Reflection on correlation between MSCI and EIMI indexes. Are we really diversificating?
     - Linear regression and prediction of MSCI World based on SP500's price.
     - Deeplearning with torch: MLP as simple neural network and LSTM for time series.
 - Portfolio simulation: 
     - Lump sum investment and dollar coast averaging with performances
     - Financial performance metrics: CAGR, ROI and sharpe ratio
   
