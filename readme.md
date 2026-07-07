# **Financial Analytics - Stock Market**
## **Overview**
An end-to-end data platform that orchestrates a robust ETL pipeline to ingest, clean, and consolidate heterogeneous financial time-series data. The system transforms raw multi-source datasets into high-integrity feature tables, feeding advanced PyTorch deep learning architectures (NHITS, LSTM, MLP) deployed as live, scalable web applications and APIs.

## **Data Pipeline & Preprocessing**
- I engineered a robust, end-to-end ETL (Extract, Transform, Load) pipeline using Python and Pandas to ingest and condition multi-source financial time-series data:
    - **Multi-Source Data Ingestion:**
        - Built a decoupled ingestion layer to process 6 heterogeneous financial datasets (VWCE, S&P 500, NASDAQ, Apple, MSCI, EIMI) from distinct CSV sources.
    - **Data Cleansing & Schema Standardization:**
        - **Time-Series Alignment:** Automated date-format standardization (handling day-first anomalies) and implemented chronological sorting with index-based time mapping to optimize downstream relational operations.
        - **String Parsing & Regex Sanitization:** Developed automated data cleaning scripts to strip percentage symbols (`%`) and remove thousands separators (`,`) from core price columns, enabling flawless float-point casting.
    - **Feature Scaling & Integration:**
        - **Data Normalization:** Integrated an iterative `MinMaxScaler` loop to standardize dynamic price ranges into an uniform scale, preventing magnitude bias during neural network training.
        - **Deterministic Merging:** Orchestrated a sequential, index-based multi-join structure using explicit schema enforcement (suffixes) to eliminate naming conflicts and output a single, high-integrity Feature Dataset.
    
## **Model Deployment & API Delivery**
- **Production-Ready Hosting:** Designed and deployed interactive cloud applications to serve model inferences as scalable APIs:
    - **Streamlit Analytics Cloud:** Deployed the end-to-end NHITS forecasting platform as a live interactive dashboard providing real-time data visualization and probabilistic analysis. 🔗 [Live Dashboard](https://financial-analytics-lwhmddpjchajmpnxgfv2gm.streamlit.app/)
    - **Hugging Face Spaces:** Containerized and hosted the core PyTorch MLP inference pipeline as a standalone web API, decoupling backend asset predictions from client interfaces. 🔗 [Live API](https://huggingface.co/spaces/ChristianChimi/Apple-Stock-API)

## **Deep Learning Modeling**
- I trained and optimized three distinct neural network architectures using PyTorch and NeuralForecaster:
    - **Robust Financial Forecasting (NHITS):**
        - Developed an advanced forecasting pipeline using the NHITS architecture to predict market trends with a 30-day horizon.
        - Integrated multi-quantile loss functions (10%, 50%, 90%) to provide probabilistic uncertainty intervals and risk assessment.
        - Optimized accuracy by incorporating dynamic exogenous variables, including VIX, S&P 500, and volume metrics.
    - **Price Prediction with PyTorch (MLP):**
        - Engineered a Multi-Layer Perceptron to evaluate cross-asset correlations, predicting target asset prices based on S&P 500 and NASDAQ index inputs.
    - **MSCI World Index Tracking (LSTM):**
        - Implemented a Long Short-Term Memory (LSTM) recurrent network to capture sequential patterns and temporal dependencies in global market time-series.
        - Implemented rigorous training monitoring by tracking and plotting loss vs. validation curves across training epochs to actively prevent overfitting.
        - Conducted comprehensive hyperparameter tuning (hidden layers, learning rate, dropout, and batch configurations) to minimize validation error.

## **Technologies Used**
 - **Python**, **Pandas**, **Numpy**, **PyTorch**, **Scikit-learn**, **NeuralForecaster**
