# DataLyst – An Interactive Data Analysis & Visualization Platform

DataLyst is a Streamlit-based interactive data analysis and visualization platform designed for exploratory data analysis (EDA), machine learning experimentation, forecasting, anomaly detection, and feature selection — all in one place.

This project is developed as a **Minor Project** and is suitable for academic submission and demonstrations.

---

## 🚀 Features

### 📂 Data Upload & Cleaning
- Upload CSV or Excel files
- Automatic handling of large datasets (row limiting for performance)
- Missing value handling:
  - Drop rows
  - Fill with mean
  - Fill with zero
- Column selection for analysis

### 📊 Exploratory Data Analysis (EDA)
- Dataset preview (head/tail)
- Statistical summary
- Correlation heatmap
- Interactive visualizations:
  - Scatter plots
  - Bar charts
  - Line charts

### 🎯 Feature Selection
- Model-based feature selection using:
  - Linear Regression
  - Decision Tree
  - Random Forest
- Recursive Feature Elimination (RFE)
- Feature importance ranking
- Human-readable explanations for why features were selected or dropped

### 🤖 Prediction & Forecasting
- Time-series forecasting using:
  - Prophet (if available)
  - ARIMA (statsmodels) as fallback
- Interactive forecasting graphs
- Confidence intervals

### 🚨 Anomaly Detection
- Isolation Forest
- DBSCAN clustering
- Scatter matrix and 2D cluster plots

### ⚡ Performance Safeguards
- Automatic row limiting for large datasets
- Safe handling of datetime and categorical data

---

## 🛠️ Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly, Matplotlib
- Prophet (optional)
- Statsmodels

---

## ▶️ How to Run

```bash
git clone https://github.com/SubhansuPradhan/DataLyst-An-Intractive-Data-analysis-Visualization-Platform-
cd DataLyst-An-Intractive-Data-analysis-Visualization-Platform-
pip install -r requirements.txt
streamlit run datalyst.py
```

---

##👤 Author

Subhansu Pradhan
B.Tech CSE (Data Science)
GIET University, Gunupur
