import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "student_performance.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
X = df.drop(columns=["final_score"])
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

linear = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)

linear.fit(X_train, y_train)
rf.fit(X_train, y_train)

def metrics(model):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    return mae, mse, rmse, r2

lm = metrics(linear)
rm = metrics(rf)

st.title("🎓 Student Performance Prediction")
st.write("Machine Learning based prediction of a student's final academic score.")

st.sidebar.header("Student Details")
hours = st.sidebar.slider("Hours Studied / Day", 0.0, 12.0, 5.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 75)
previous = st.sidebar.slider("Previous Score", 0, 100, 60)
assignments = st.sidebar.slider("Assignments Completed", 0, 10, 7)
sleep = st.sidebar.slider("Sleep Hours / Day", 3.0, 10.0, 7.0, 0.5)
extra = st.sidebar.selectbox("Extracurricular Activity", ["No", "Yes"])
extra_value = 1 if extra == "Yes" else 0

input_data = pd.DataFrame([{
    "hours_studied": hours,
    "attendance_percent": attendance,
    "previous_score": previous,
    "assignments_completed": assignments,
    "sleep_hours": sleep,
    "extracurricular": extra_value
}])

pred_linear = float(linear.predict(input_data)[0])
pred_rf = float(rf.predict(input_data)[0])
final_pred = (pred_linear + pred_rf) / 2
final_pred = max(0, min(100, final_pred))

col1, col2, col3 = st.columns(3)
col1.metric("Linear Regression", f"{pred_linear:.2f}")
col2.metric("Random Forest", f"{pred_rf:.2f}")
col3.metric("Final Prediction", f"{final_pred:.2f}/100")

if final_pred >= 75:
    status = "High Performance"
elif final_pred >= 50:
    status = "Moderate Performance"
else:
    status = "Needs Academic Support"

st.success(f"Predicted Status: **{status}**")

st.subheader("Model Evaluation")
evaluation = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lm[0], rm[0]],
    "MSE": [lm[1], rm[1]],
    "RMSE": [lm[2], rm[2]],
    "R2 Score": [lm[3], rm[3]]
})
st.dataframe(evaluation.style.format({
    "MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.3f}"
}), use_container_width=True)

st.subheader("Dataset Preview")
st.dataframe(df, use_container_width=True)

st.subheader("How the project works")
st.markdown("""
**Dataset → Preprocessing → Train/Test Split → Model Training → Evaluation → Prediction**

- Linear Regression predicts the score using a linear relationship between features and final score.
- Random Forest uses multiple decision trees and combines their predictions.
- MAE, MSE, RMSE and R² are used to evaluate regression performance.
""")
