# Resale-Car-value-predictor
🚗 An AI-driven Used Car Price Predictor built with Python, XGBoost, and Flask. Scraped from PakWheels data, it utilizes an optimized pipeline (IQR cleaning, StandardScaler, Label Encoding) to deliver real-time automotive market valuations alongside a dark-themed data analytics dashboard dashboard! 📊


# 🚗 Resale Car Value Predictor (PakWheels AI)

An interactive, full-stack Machine Learning application designed to predict the market valuation of used vehicles in Pakistan. Powered by an optimized **XGBoost Regressor** pipeline on the backend and deployed via a responsive **Flask** web application, this system handles everything from text-based data extraction to real-time predictive serving and market analytics.

---

## 🚀 Key Features

*   **Dual-Model Backend Engine:** Trains and evaluates both **Linear Regression** and highly tuned **XGBoost Regressor** frameworks to ensure maximum predictive accuracy.
*   **Real-Time API Inference:** An interactive Flask web application that serves instant evaluation predictions (in Lakhs/Lacs) based on specialized drop-down vehicle metrics.
*   **Dynamic Data Visualizations:** A dark-themed analytics dashboard interface rendering critical real-world automotive market insights, including:
    *   Actual vs. Predicted Price performance plots.
    *   Market Price Frequency Distribution densities.
    *   Average Car Valuations segmented by Vehicle Body Types.
    *   Macro price trajectory trends across Manufacturing Years.
*   **Robust ETL Pipeline:** Fully automated data wrangling scripts that clean, impute, and extract structured metrics out of dirty scraped e-commerce web values.

---

## 🛠️ System Architecture & Data Pipeline

The data layer utilizes localized market insights scraped from **PakWheels.com**, mapping out critical real-world vehicle attributes.

### 1. Data Cleansing & Text Parsing
*   **Custom Regex Transformers:** Converted dirty string values (e.g., `"PKR 23.9 lacs"`, `"1.2 Crore"`, `"120,000 km"`, `"660 cc"`) into analytical floating-point numerical vectors.
*   **Geographic Mapping:** Normalized fractured provincial structures into localized metropolitan trade hubs (`Lahore`, `Karachi`, `Islamabad`, `Peshawar`, `Quetta`, `Faisalabad`, `Multan`).
*   **Outlier Purging:** Implemented **Interquartile Range (IQR)** filtering blocks across numeric variables (`Price`, `Millage`, `EngineCapacity`) to eliminate market anomalies and skewed listings.

### 2. Feature Engineering & Preprocessing
*   **Categorical Encoding:** Leveraged `LabelEncoder` to translate structural fields (`Fuel`, `Transmission`, `Assembly`, `BodyType`, `Color`, `City`) into numerical indicators.
*   **Feature Scaling:** Deployed a `StandardScaler` pipeline to eliminate scaling biases across skewed parameters like odometer readings (`Millage`) and manufacturing years.

---

## 📊 Model Performance

Upon standalone training execution, the backend evaluates performance metrics over independent verification test splits:

| Evaluation Metric | Linear Regression | XGBoost Regressor (Selected) |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | Evaluating... | **Optimized / Lowest** |
| **Root Mean Squared Error (RMSE)**| Evaluating... | **Optimized / Lowest** |
| **R² Score (Variance Explained)** | Baseline | **Highest Performance** |

---

## 💻 Tech Stack Used

*   **Machine Learning Core:** Python, Pandas, NumPy, Scikit-Learn, XGBoost
*   **Data Visualization Backend:** Matplotlib, Seaborn
*   **Web Framework & Server Integration:** Flask (Python Server Engine), JSON-IPC API REST Endpoints
*   **Frontend Layout Layer:** HTML5, CSS3, JavaScript (Asynchronous `fetch` payload handling)

---

## ⚙️ How to Setup and Run Local Server

### 1. Install System Dependencies
Ensure you have Python installed, then clone the repository and fetch the required packages via terminal:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask
