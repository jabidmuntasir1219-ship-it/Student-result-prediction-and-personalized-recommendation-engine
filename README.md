# Time-Aware Personalized Student Blueprint & Recommendation Engine

A machine learning prototype that predicts a student’s **next-week academic performance** from recent academic and behavioral inputs, then generates **personalized recommendations** using what-if simulation and explainable AI techniques.

---

## Overview

This project is a **time-aware personalized student performance prediction system** built with Python and scikit-learn.  
It uses synthetic longitudinal student data to train a **RandomForestRegressor** that predicts the next week’s score based on a student’s current academic state.

The system also includes:
- **Explainability** with SHAP when available.
- A **fallback feature-importance layer** when SHAP is not installed.
- A **personalized recommendation engine** that simulates changes in study habits and predicts the effect on performance.
- **Validation checks** for user input.
- **Metric logging** to save model evaluation results.

---

## Key Features

- Synthetic longitudinal student dataset generation.
- Time-aware prediction of next-week academic score.
- Student-level data split using `GroupShuffleSplit`.
- Random Forest regression model.
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- SHAP-based interpretability support.
- Fallback explanation using feature importance if SHAP is unavailable.
- Interactive CLI-based input system.
- Personalized what-if simulation and recommendation output.
- Metrics saved to `metrics.json`.

---

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- SHAP
- JSON

---

## Engine Architecture

```mermaid
graph TD
A[Synthetic Longitudinal Data Generator] --> B[Time-Aware Feature Engineering and Target Alignment]
B --> C[GroupShuffleSplit Leakage Control]
C --> D[Training Set]
C --> E[Isolated Test Set]
D --> F[RandomForest Regressor]
F --> G[Production Logs and metrics.json Generation]
F --> H[Interactive User Input Interface]
H --> I[Explainability Layer Dynamic SHAP or Fallback Engine]
I --> J[Prescriptive Counterfactual Engine What-If Scenario Optimization]
J --> K[Final Personalized Blueprint]
