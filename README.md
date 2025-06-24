# 🧪 ML Regression on Raman SERS Data

This project applies and compares various regression algorithms to **predict chemical concentrations** from Raman SERS (Surface Enhanced Raman Spectroscopy) data. It aims to evaluate model performance and identify the most suitable regression approach for spectral signal interpretation.

---

## 📌 Objective

To **analyze and compare** the performance of multiple machine learning regression models in predicting target concentration values from Raman spectral data. This helps in enhancing analytical chemistry methods using computational models.

---

## 📚 About the Dataset

- **Source**: Simulated or experimental SERS data (CSV format)
- **Features**: Spectral intensities, peak positions, etc.
- **Target**: Chemical concentration (`Conc`)

---

## 🔍 Regression Algorithms Used

Each model is implemented in its own Jupyter Notebook:

| Model Type         | Description                                |
|--------------------|--------------------------------------------|
| Linear Regression  | Baseline model using simple linear mapping |
| Polynomial Regression | Captures non-linearity by expanding features |
| Lasso Regression (L1) | Performs feature selection & regularization |
| Ridge Regression (L2) | Penalizes large coefficients to avoid overfitting |

---

## ⚙️ Workflow Summary

1. **Data Preprocessing**
   - CSV loading
   - One-hot encoding (if needed)
   - Scaling features using `StandardScaler`
   - Train-test split (80-20)

2. **Model Training & Evaluation**
   - Fit model on training data
   - Predict test outputs
   - Evaluate using R², MAE, RMSE
   - Visualize Actual vs Predicted values

3. **Model Comparison**
   - Metrics summarized in a comparison table
   - Interpretation of which model performed best

---

## 📈 Evaluation Metrics

For each model, the following metrics are computed:

- **R² Score** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

---

## 📊 Visualizations

Each notebook includes a scatter plot of:
