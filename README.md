# Time-Aware Personalized Student Blueprint & Recommendation Engine

An enterprise-grade, longitudinal machine learning pipeline designed to track, explain, and optimize student performance over time. This repository implements a robust prescriptive analytics engine that moves beyond static descriptive metrics. It predicts a student's next-week score based on sequential behavioral features, leverages game-theoretic explanations (SHAP), and executes real-time counterfactual "What-If" simulations to output highly personalized action plans.

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

---------------------------------------+
| Synthetic Longitudinal Data Generator |
+---------------------------------------+
|
v
+---------------------------------------+
|   Time-Aware Feature Engineering &     |
|   Target Alignment (Shift Logic)      |
+---------------------------------------+
|
v
+---------------------------------------+
|  GroupShuffleSplit (Leakage Control)  |
+---------------------------------------+
/                     
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
---

## 📋 Features Monitored (14 Core Dimensions)

The model utilizes 14 distinct behavioral, psychological, and performance metrics:
1. `study_hours` (Weekly)
2. `attendance` (Percentage)
3. `sleep_hours` (Nightly Average)
4. `screen_time` (Daily Average)
5. `homework_completion` (Percentage)
6. `previous_score` (Historical Anchor)
7. `class_test_avg` (Immediate Performance)
8. `attendance_trend` (Binary Directional)
9. `study_consistency` (Binary Tracker)
10. `stress_level` (Categorical Scale 1-5)
11. `class_participation` (Scale 1-5)
12. `revision_frequency` (Weekly Cycles)
13. `assignment_delay` (Count)
14. `weekend_study` (Hours)

---

## 💻 Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ installed.

### Dependencies
Install the required production libraries via `pip`:
```bash
pip install pandas numpy scikit-learn
