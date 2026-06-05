import pandas as pd
import numpy as np
import json
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("[SYSTEM NOTICE] SHAP library not found. Falling back to internal feature importance layer.")

print("==================================================================")
print("PRODUCTION SYSTEM: Time-Aware Personalized Blueprint Engine Engaged.")
print("==================================================================\n")

np.random.seed(42)
num_students = 500
num_weeks = 12

features = [
    "study_hours", "attendance", "sleep_hours", "screen_time",
    "homework_completion", "previous_score", "class_test_avg",
    "attendance_trend", "study_consistency", "stress_level",
    "class_participation", "revision_frequency", "assignment_delay",
    "weekend_study", "historical_test_avg"
]

raw_data = []
for sid in range(1, num_students + 1):
    base_study = np.random.randint(3, 12)
    base_sleep = np.random.randint(4, 9)
    base_screen = np.random.uniform(2, 8)
    prev_score = np.random.randint(50, 85)
    
    running_test_scores = []

    for week in range(1, num_weeks + 1):
        study_hours = float(np.clip(base_study + np.random.randint(-2, 3), 2, 16))
        sleep_hours = float(np.clip(base_sleep + np.random.randint(-1, 2), 3, 10))
        screen_time = float(np.clip(base_screen + np.random.uniform(-1, 1), 1, 12))
        attendance = float(np.random.randint(50, 100))
        homework_completion = float(np.random.randint(20, 100))
        class_test_avg = float(np.random.randint(40, 100))
        
        running_test_scores.append(class_test_avg)
        historical_test_avg = float(np.mean(running_test_scores))
        
        attendance_trend = int(np.random.choice([0, 1]))
        study_consistency = int(np.random.choice([0, 1]))
        stress_level = int(np.random.randint(1, 6))
        class_participation = int(np.random.randint(1, 6))
        revision_frequency = int(np.random.randint(0, 5))
        assignment_delay = int(np.random.randint(0, 6))
        weekend_study = float(np.random.randint(0, 8))

        weekly_score = (
            (study_hours * 1.5) + (attendance * 0.3) + (prev_score * 0.2) +
            (sleep_hours * 1.0) + (homework_completion * 0.1) - (screen_time * 1.5) +
            (class_test_avg * 0.1) + (attendance_trend * 2.0) + (study_consistency * 2.0) -
            (stress_level * 1.2) + (class_participation * 0.8) + (revision_frequency * 1.0) -
            (assignment_delay * 1.5) + (weekend_study * 0.5) + (historical_test_avg * 0.05)
        )
        weekly_score = int(np.clip(weekly_score + np.random.normal(0, 2), 0, 100))

        raw_data.append([
            sid, week, study_hours, attendance, sleep_hours, screen_time,
            homework_completion, prev_score, class_test_avg, attendance_trend,
            study_consistency, stress_level, class_participation,
            revision_frequency, assignment_delay, weekend_study, historical_test_avg, weekly_score
        ])
        prev_score = weekly_score 

cols = ["student_id", "week"] + features + ["current_score"]
df = pd.DataFrame(raw_data, columns=cols)

df['next_week_score'] = df.groupby('student_id')['current_score'].shift(-1)
df = df.dropna()

X = df[features]
y = df['next_week_score']
student_groups = df['student_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=student_groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
metrics_log = {
    "Model": "RandomForestRegressor_TimeAware",
    "Test_R2_Score": round(r2_score(y_test, test_predictions), 4),
    "Test_MAE": round(mean_absolute_error(y_test, test_predictions), 4),
    "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, test_predictions)), 4)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=4)

tree_explainer = shap.Explainer(model, X_train) if SHAP_AVAILABLE else None

def generate_live_report(user_features, input_score, model, explainer=None):
    current_state_df = pd.DataFrame([user_features])[features]
    baseline_pred = model.predict(current_state_df)[0]

    print("\n" + "="*60)
    print("                YOUR PERSONALIZED DASHBOARD             ")
    print("============================================================")
    print(f"Current Week's Base Score: {input_score}")
    print(f"Predicted Score for Next Week (Without Changes): {baseline_pred:.1f}\n")

    if explainer and SHAP_AVAILABLE:
        print("[>] Diagnostic Layer (SHAP Dynamic Impact Analysis):")
        shap_values = explainer(current_state_df)
        contributions = pd.DataFrame({
            'Feature': features,
            'Contribution': shap_values.values[0]
        }).sort_values(by='Contribution', key=abs, ascending=False)
        
        for _, row in contributions.head(3).iterrows():
            impact = "positive (+)" if row['Contribution'] > 0 else "negative (-)"
            print(f"    * {row['Feature']}: {abs(row['Contribution']):.2f} points {impact} impact.")
    else:
        print("[>] Diagnostic Layer (Internal Feature Importance Fallback Engine):")
        importances = model.feature_importances_
        fallback_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("    (Using trained model weights to isolate top driver attributes for your profile)")
        for _, row in fallback_df.head(3).iterrows():
            print(f"    * {row['Feature']}: Global feature importance influence weight is {row['Importance']*100:.1f}%.")
    print()

    print("[>] Prescriptive Recommendations (What-If Simulations):")
    scenarios = {
        "Increase study hours by 2": {"study_hours": min(16, user_features["study_hours"] + 2)},
        "Improve attendance to 95%": {"attendance": max(95, user_features["attendance"])},
        "Sleep strictly 8 hours": {"sleep_hours": 8},
        "Reduce screen time by 2 hours": {"screen_time": max(1, user_features["screen_time"] - 2)},
        "Revise 2 more times": {"revision_frequency": min(5, user_features["revision_frequency"] + 2)}
    }

    best_action = "Maintain current routine"
    max_improvement = 0

    for action_name, tweaked_feature in scenarios.items():
        simulated_state = user_features.copy()
        simulated_state.update(tweaked_feature)

        simulated_df = pd.DataFrame([simulated_state])[features]
        simulated_pred = model.predict(simulated_df)[0]
        improvement = simulated_pred - baseline_pred

        if improvement > max_improvement:
            max_improvement = improvement
            best_action = action_name
            
        sign = "+" if improvement > 0 else ""
        print(f"    {sign}{improvement:.1f} pts -> If you: {action_name}")

    print(f"\n>> MACHINE LEARNING OPTIMAL DIRECTIVE: {best_action}")
    print(f">> EXPECTED NEXT WEEK BOOST: +{max_improvement:.1f} points")
    print("="*60 + "\n")

def get_validated_input(prompt, min_val, max_val, is_int=False):
    while True:
        try:
            val_type = "Integer" if is_int else "Float/Number"
            user_in = input(f"{prompt} ({val_type} between {min_val} and {max_val}): ")
            val = int(user_in) if is_int else float(user_in)
            if min_val <= val <= max_val:
                return val
            print(f"    [Validation Error] Input must be strictly between {min_val} and {max_val}. Try again.")
        except ValueError:
            print(f"    [Type Error] Invalid data type. Please enter a valid numerical value.")

def run_interactive_session():
    print("Enter your exact performance data for the current week:")
    
    input_score = get_validated_input("1. Current Week Final Score", 0, 100, is_int=True)
    
    inputs = {}
    inputs["study_hours"] = get_validated_input("2. Study Hours per week", 2, 16)
    inputs["attendance"] = get_validated_input("3. Attendance Rate Percentage", 40, 100)
    inputs["sleep_hours"] = get_validated_input("4. Average Sleep Hours", 3, 10)
    inputs["screen_time"] = get_validated_input("5. Daily Screen Time Hours", 1, 12)
    inputs["homework_completion"] = get_validated_input("6. Homework Completed %", 10, 100)
    inputs["previous_score"] = float(input_score)
    inputs["class_test_avg"] = get_validated_input("7. Class Test Average", 40, 100)
    inputs["attendance_trend"] = get_validated_input("8. Attendance Trend (1=Improving, 0=Declining)", 0, 1, is_int=True)
    inputs["study_consistency"] = get_validated_input("9. Study Consistency (1=Consistent, 0=Inconsistent)", 0, 1, is_int=True)
    inputs["stress_level"] = get_validated_input("10. Current Stress Level (1-5)", 1, 5, is_int=True)
    inputs["class_participation"] = get_validated_input("11. Class Participation Score (1-5)", 1, 5, is_int=True)
    inputs["revision_frequency"] = get_validated_input("12. Weekly Revision Count", 0, 5, is_int=True)
    inputs["assignment_delay"] = get_validated_input("13. Number of Delayed Assignments", 0, 6, is_int=True)
    inputs["weekend_study"] = get_validated_input("14. Weekend Study Hours", 0, 8)
    
    inputs["historical_test_avg"] = (inputs["class_test_avg"] + inputs["previous_score"]) / 2

    generate_live_report(inputs, input_score, model, tree_explainer)

run_interactive_session()
