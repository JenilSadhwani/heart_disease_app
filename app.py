import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from utils.report_gen import get_dynamic_response_gemini, generate_health_report_pdf
import os
from sklearn.impute import SimpleImputer

# === Load Models ===
model_files = {
    "Random Forest": "models/RAF.pkl",
    "Logistic Regression": "models/LOR (2).pkl",
    "Decision Tree": "models/DT (2).pkl",
    "SVM": "models/SVM (1).pkl",
    "ANN": "models/ANN.keras",
    "FNN": "models/FNN.keras",
    "GRU": "models/GRU.keras",
    "LSTM": "models/LSTM.keras"
}

models = {}
for name, path in model_files.items():
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        elif path.endswith(".keras") or path.endswith(".h5"):
            models[name] = load_model(path, custom_objects={"Orthogonal": Orthogonal})
    except Exception as e:
        print(f"Could not load {name}: {e}")

# === Optional Scaler ===
scaler_path = "models/scaler.pkl"
sc_x = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None

# === Prediction Function ===
def predict_heart_disease(
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, model_choice
):
    inputs = {
        "Age": age,
        "Sex": sex,
        "CP": cp,
        "trestbps": trestbps,
        "Chol": chol,
        "FBS": fbs,
        "restECG": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "Ca": ca
    }

    input_array = np.array([list(inputs.values())])
    if np.isnan(input_array).any():
        return "❌ Missing input(s)", "Please make sure all fields are filled.", None

    if sc_x:
        input_array = sc_x.transform(input_array)

    model = models.get(model_choice)
    if not model:
        return "Model not found", "Model not found", None

    pred_raw = model.predict(input_array)
    prediction = int(pred_raw[0]) if pred_raw.shape == (1,) else int(pred_raw[0][0] >= 0.5) if pred_raw.shape == (1, 1) else int(np.argmax(pred_raw[0]))

    risk = "High Risk" if prediction == 1 else "Low Risk"
    advice = get_dynamic_response_gemini(prediction, inputs)
    filename = f"Heart_Report_{model_choice}.pdf"
    generate_health_report_pdf(inputs, prediction, advice, filename)

    return risk, advice, filename

# === Streamlit UI ===
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor + Gemini AI Advice")
st.markdown("Quickly evaluate heart disease risk and receive personalized AI-generated health guidance.")

with st.form("input_form"):
    st.subheader("📝 Patient Medical Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.radio("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression", value=1.0)
    slope = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3])
    model_choice = st.selectbox("Choose Model", options=list(models.keys()))
    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    risk, advice, pdf_file = predict_heart_disease(
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, model_choice
    )

    st.subheader("🔎 Risk Assessment")
    st.success(risk)

    st.subheader("💬 Gemini AI Health Advice")
    st.markdown(advice)

    if pdf_file:
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=pdf_file, mime="application/pdf")
