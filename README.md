# 📊 Pharmaceutical Sales Forecasting Using Stacking Ensemble Models

## 🔍 Problem Statement

Traditional pharmaceutical sales forecasting struggles with multiple drug categories and non-linear trends. This leads to stockouts or overstocking, adversely affecting patient access and supply chain efficiency.

## 🎯 Objective

To develop a robust machine learning model using **stacking ensemble** techniques (CatBoost, XGBoost, Random Forest) for accurate multi-category drug demand forecasting.

## 🧠 Methodology

- **Data Collection**: Real-world multi-year sales across 8 drug categories.
- **Preprocessing**: Missing value handling, normalization, categorical encoding.
- **Modeling**:
  - **Base models**: CatBoost and XGBoost
  - **Meta-model**: Random Forest using base predictions
- **Evaluation**: Metrics including MAE, MSE, MAPE, R²

| Model      | MSE    | MAE  | MAPE (%) | Accuracy (%) |
|------------|--------|------|-----------|----------------|
| CatBoost   | 123.38 | 8.47 | 12.6      | 87.4           |
| XGBoost    | 118.32 | 8.13 | 11.7      | 88.3           |
| **Stacking** | **103.75** | **6.96** | **9.9** | **90.1**         |

## 📊 Visualization

- Time-series plots and trend analysis
- Confusion matrix for categorical predictions
- Feature importance plots

## 🚧 Limitations

- Model trained only on historical sales & temporal features
- Model interpretability is limited
- Ensemble architecture is static

## 🔮 Future Enhancements

- Include exogenous data (weather, policy changes, epidemics)
- Explore deep learning (LSTM, Transformers)
- Enhance explainability with SHAP/LIME
- Real-time model deployment via APIs

## 📁 Project Structure

