# Machine Learning Regression on Raman SERS Spectroscopy Data

## Project Summary

This project explores the use of advanced regression algorithms to predict chemical analyte concentrations from high-dimensional Raman SERS (Surface-Enhanced Raman Spectroscopy) data. By combining traditional linear models with ensemble learning techniques, we aim to benchmark predictive performance and robustness across a diverse model suite.

The ultimate goal is to develop scalable, noise-tolerant regression workflows that can be deployed in real-world spectroscopic analysis — particularly in analytical chemistry, pharmaceuticals, and environmental sensing.

---

## Objective

To evaluate and compare a suite of machine learning regression algorithms for their ability to learn from Raman spectral data and predict chemical concentrations accurately. The models are assessed on performance, generalization, and interpretability, under realistic data preprocessing conditions.

---

## Dataset Description

- **Source**: Simulated or experimental Raman SERS spectra
- **Format**: CSV
- **Features**: Raman intensity values across discrete wavenumber shifts
- **Target**: Analyte concentration (`Conc`)

Each sample corresponds to one Raman spectrum with thousands of features representing molecular vibrational signatures.

---

## Implemented Models

| Model                        | Description                                                         |
|-----------------------------|----------------------------------------------------------------------|
| Linear Regression           | Baseline linear mapping for comparison                              | 
| Polynomial Regression       | Captures nonlinear behavior via polynomial feature expansion         | 
| Lasso Regression (L1)       | L1 regularization for feature selection and sparsity                 | 
| Ridge Regression (L2)       | L2 regularization for reducing model complexity                      | 
| Random Forest Regressor     | Bagged decision trees for variance reduction and robustness          | 
| AdaBoost Regressor          | Boosted weak learners with sequential error correction               | 
| Gradient Boosting Regressor | Optimized additive model using gradient descent                      | 
| Gradient Boosting + ES      | Gradient Boosting with Early Stopping for overfitting control        | 

All models follow a standardized pipeline for preprocessing, training, and evaluation.

---

## Workflow Summary

1. Load spectral dataset from CSV
2. Preprocess data (scaling, normalization, feature checks)
3. Split into training and test sets (typically 80/20)
4. Train each regression model
5. Predict on test data
6. Evaluate model using standardized metrics
7. Generate visual comparisons (Actual vs Predicted, Residuals)
8. Compile comparison table

---

## Evaluation Metrics

- **R² Score (Coefficient of Determination)**  
  Indicates how well the model explains the variance in target values.

- **MAE (Mean Absolute Error)**  
  Measures the average absolute difference between predicted and true concentrations.

- **RMSE (Root Mean Squared Error)**  
  Penalizes larger errors more heavily; sensitive to outliers.

All metrics are reported for each model to enable quantitative comparison.

---

## Visualizations and Outputs

Each model notebook includes:

- Predicted vs Actual scatter plots
- Residual error distribution
- Feature importance plots (for tree-based models)
- Metric tables for model benchmarking

These visualizations provide critical insight into predictive accuracy and model behavior.

---

## Technology Stack

- Python 3.10+
- Jupyter Notebooks
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for all models and evaluation tools
- Optional: `xgboost`, `lightgbm` (for future enhancements)

---
## Model Performance Comparison

| Model                            | R² Score | MAE         | RMSE       |
|----------------------------------|----------|-------------|------------|
| Linear Regression                | -0.25    | 0.0532      | 0.1443     |
| Polynomial Regression            | 0.8325   | 0.0134      | 0.0528     |
| Lasso Regression                 | 0.9836   | 13660.0460  | 21598.8228 |
| Ridge Regression                 | 0.8746   | 0.0138      | 0.0457     |
| Bagging Regressor               | 0.8376   | 0.0083      | 0.0520     |
| Random Forest Regressor         | 1.0000   | 50.9979     | 341.4244   |
| Gradient Boosting Regressor     | 1.0000   | 4.4656      | 11.6348    |
| AdaBoost Regressor              | 1.0000   | 88.8531     | 180.1800   |
| Gradient Boosting + Early Stop  | 1.0000   | 0.8875      | 8.1836     |

---
## Future Enhancements

- Incorporate dimensionality reduction (PCA, t-SNE) for visualization and modeling
- Expand to deep learning models (e.g., CNN for raw spectra)
- Implement robust cross-validation with hyperparameter tuning
- Evaluate model robustness under synthetic noise conditions
- Deploy as a web-based spectroscopy analysis tool using Streamlit

---

## Why It Matters

Surface-Enhanced Raman Spectroscopy is a highly sensitive analytical technique, but decoding spectral data at scale remains a challenge. By applying machine learning regression to this problem, we aim to automate and enhance chemical quantification processes, enabling rapid, reproducible, and intelligent analytics in domains ranging from pharmaceuticals to materials science.

---

## References

1. Z. Al‐Shaebi, M. Akdeniz, A. O. Ahmed, M. Altunbek, and O. Aydin,  
   *Breakthrough solution for antimicrobial resistance detection: Surface‐Enhanced RAMAN spectroscopy‐based on artificial intelligence*,  
   **Advanced Materials Interfaces**, Nov. 2023.  
   [DOI: 10.1002/admi.202300664](https://doi.org/10.1002/admi.202300664)

2. J. Nie, G. Zhang, X. Lu, H. Wang, C. Sheng, and L. Sun,  
   *Obstacle avoidance method based on reinforcement learning dual-layer decision model for AGV with visual perception*,  
   **Control Engineering Practice**, vol. 153, p. 106121, Oct. 2024.  
   [DOI: 10.1016/j.conengprac.2024.106121](https://doi.org/10.1016/j.conengprac.2024.106121)

3. X. Bi, L. Lin, Z. Chen, and J. Ye,  
   *Artificial intelligence for Surface‐Enhanced Raman Spectroscopy*,  
   **Small Methods**, vol. 8, no. 1, Oct. 2023.  
   [DOI: 10.1002/smtd.202301243](https://doi.org/10.1002/smtd.202301243)

4. D. Cialla et al.,  
   *Surface-enhanced Raman spectroscopy (SERS): progress and trends*,  
   **Analytical and Bioanalytical Chemistry**, vol. 403, no. 1, pp. 27–54, Dec. 2011.  
   [DOI: 10.1007/s00216-011-5631-x](https://doi.org/10.1007/s00216-011-5631-x)
