# Time-Aware Personalized Student Blueprint & Recommendation Engine

An enterprise-grade, longitudinal machine learning pipeline designed to track, explain, and optimize student performance over time. This repository implements a robust prescriptive analytics engine that moves beyond static descriptive metrics. It predicts a student's next-week score based on sequential behavioral features, leverages game-theoretic explanations (SHAP), and executes real-time counterfactual "What-If" simulations to output highly personalized action plans.

> ⚠️ **Data Source Disclaimer**: This repository functions out-of-the-box as a **Synthetic Demo/Prototype** using algorithmically simulated student life-logging patterns. While the pipeline architecture is production-ready, testing it against real-world human behaviors requires linking it to authentic institutional datasets or unified student logs.

---

## 🚀 Key Architectural Upgrades

This project addresses and corrects several classic flaws found in tutorial-level machine learning implementations:

* **Leakage-Free Validation (`GroupShuffleSplit`):** Traditional random train/test splits corrupt longitudinal datasets by spreading a single student's chronological history across both train and test partitions (causing artificial memorization). This pipeline enforces strict isolation by grouping records by `student_id`.
* **Time-Aware Context Extraction:** To capture behavioral patterns over a shifting timeline, the engine synthesizes rolling feature states (`historical_test_avg`) alongside forward-shifted regression targets ($Week \ t \rightarrow Week \ t+1$).
* **Dual-Layer Explainability (SHAP + Dynamic Fallback):** Integrates Shapley Additive exPlanations to diagnose the exact root causes behind a student's baseline prediction. If the heavy `shap` library is missing in production, a mathematical fallback layer instantly activates using internal model weights to prevent runtime crashes.
* **Prescriptive Counterfactual Engine:** Instead of executing fragile, hardcoded logical rules (e.g., `if sleep < 6`), the engine runs multiple concurrent simulations using the trained model to find the single behavior alteration yielding the maximum mathematical score increase.
* **Production-Grade Guardrails:** Features strict, type-safe range validation on all real-time terminal inputs to fully eliminate out-of-bounds anomalies or unexpected system crashes.

---

## 🛠️ System Architecture

```text
  +---------------------------------------+
  | Synthetic Longitudinal Data Generator |
  +---------------------------------------+
                      |
                      v
  +---------------------------------------+
  |   Time-Aware Feature Engineering &    |
  |   Target Alignment (Shift Logic)      |
  +---------------------------------------+
                      |
                      v
  +---------------------------------------+
  |  GroupShuffleSplit (Leakage Control)  |
  +---------------------------------------+
           /                     \
          v                       v
  +---------------+       +---------------+
  | Training Set  |       | Isolated Test |
  +---------------+       +---------------+
          |                       |
          v                       v
  +---------------+       +----------------------------+
  | RandomForest  |------>| Production Logs            |
  |   Regressor   |       | (metrics.json Generation)  |
  +---------------+       +----------------------------+
          |
          v
  +----------------------------------+
  | Interactive User Input Interface |
  +----------------------------------+
          |
          v
  +----------------------------------+
  |       EXPLAINABILITY LAYER       |
  |  (Dynamic SHAP / Fallback Engine)|
  +----------------------------------+
          |
          v
  +----------------------------------+
  | PRESCRIPTIVE COUNTERFACTUAL ENGINE|
  | ("What-If" Scenario Optimization)|
  +----------------------------------+
          |
          v
  +----------------------------------+
  |   Final Personalized Blueprint   |
  +----------------------------------+

## 🛠️ Prerequisites & Data Environment

Before setting up and executing the pipeline, ensure your environment meets the structural, operational, and data requirements defined below.

### 1. Software & Environment Constraints
* **Python Version:** `Python >= 3.8` (Required for advanced `scikit-learn` partitioning pipelines and `shap` compatibility).
* **Operating System:** Cross-platform compatibility verified across Linux, macOS, and Windows environments.

### 2. Core Dependencies
The system relies on a clean data science stack. You can install all runtime dependencies via the accompanying `requirements.txt`:
```bash
pip install pandas numpy scikit-learn

