import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from utils.report_gen import get_dynamic_response_gemini, generate_health_report_pdf

# === Define Model Paths ===
model_files = {
    "Random Forest": "models/RAF.pkl",
    "Logistic Regression": "models/LOR (2).pkl",  # Use .keras instead of .pkl
    "Decision Tree": "models/DT (2).pkl",
    "SVM": "models/SVM (1).pkl"
}

# === Load Models (supports .pkl and .keras) ===
models = {}
for name, path in model_files.items():
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        elif path.endswith(".keras") or path.endswith(".h5"):
            models[name] = load_model(path)
        else:
            st.warning(f"⚠️ Unsupported model format for: {path}")
    except Exception as e:
        st.warning(f"⚠️ Could not load {name} model from {path}: {e}")

# === Optional: Load Scaler ===
scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    sc_x = pickle.load(open(scaler_path, "rb"))
else:
    st.warning("⚠️ No scaler found. Proceeding without input normalization.")
    sc_x = None

# === Streamlit Form UI ===
st.title("❤️ Heart Disease Risk Predictor")

with st.form("heart_form"):
    st.subheader("📝 Enter Patient Details")
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
    chol = st.number_input("Serum Cholesterol (mg/dl)")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression induced by exercise")
    slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels colored", [0, 1, 2, 3])
    model_choice = st.selectbox("Choose ML Model", list(models.keys()))

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_vals = {
        "Age": age, "Sex": sex, "CP": cp, "trestbps": trestbps, "Chol": chol,
        "FBS": fbs, "restECG": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "Ca": ca
    }

    input_array = np.array([list(input_vals.values())])
    if sc_x:
        input_array = sc_x.transform(input_array)

    model = models[model_choice]
    try:
        pred_raw = model.predict(input_array)
        if pred_raw.shape == (1,):
            prediction = int(pred_raw[0])
        elif pred_raw.shape == (1, 1):
            prediction = int(pred_raw[0][0] >= 0.5)
        else:
            prediction = int(np.argmax(pred_raw[0]))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Output
    st.subheader("🧠 Prediction Result")
    risk = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"**{risk} of Heart Disease**")

    # Gemini AI Advice
    st.subheader("💬 Gemini Health Assistant Advice")
    advice = get_dynamic_response_gemini(prediction, input_vals)
    st.markdown(advice)

    # Generate PDF
    if st.button("📄 Download PDF Report"):
        filename = f"Heart_Report_{model_choice}.pdf"
        generate_health_report_pdf(input_vals, prediction, advice, filename)
        with open(filename, "rb") as f:
            st.download_button("📥 Click to Download", f, file_name=filename)
