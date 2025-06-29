# XG-Boost-Algorithm

# 🌫️ AQI Prediction using XGBoost

This project uses **XGBoost** — a powerful and scalable machine learning algorithm — to predict **Air Quality Index (AQI)** from atmospheric pollutant data. XGBoost outperforms traditional models by using advanced boosting techniques and regularization.

---

## 📌 Overview

Air pollution is a major concern, and forecasting the **Air Quality Index** (AQI) can help with proactive health and environmental measures. This project:
- Predicts AQI using features like CO, NO₂, SO₂, O₃, PM2.5, and PM10
- Implements XGBoost Regressor for better accuracy
- Evaluates performance using R², MSE, and RMSE
- Compares with Linear and Gradient Boosting regressors

---

## 📁 Dataset

- **File**: `New_York_Air_Quality.csv`
- **Features**:
  - `CO`, `NO2`, `SO2`, `O3`, `PM2.5`, `PM10` — pollutant concentrations
  - `AQI` — target value (Air Quality Index)
- **Source**: Real-world environmental data (New York)

---

## ⚙️ Technologies & Libraries

- Python 3.12
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- pandas, numpy
- scikit-learn (train/test split, evaluation metrics)

---

## 💡 Model

### ✅ XGBoost Regressor

XGBoost is an **optimized version** of Gradient Boosting with support for:
- Regularization (L1, L2)
- Handling missing values
- Parallel computation
- Fast performance with high accuracy

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)




📊 Model Performance
Metric	Value
R² Score	0.5531
MSE	105.02
RMSE	10.25

📈 XGBoost slightly outperforms both Linear and Gradient Boosting models, delivering more accurate AQI predictions.
