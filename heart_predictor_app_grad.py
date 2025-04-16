import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from utils.report_gen import get_dynamic_response_gemini, generate_health_report_pdf
import os
from sklearn.impute import SimpleImputer

# === Load Models ===
model_files = {
    "Random Forest": "models/RAF.pkl",
    "Logistic Regression": r"C:\Users\ASUS\Downloads\heart_disease_app\models\LOR (2).pkl",
    "Decision Tree": "models/DT (2).pkl",
    "SVM": "models/SVM (1).pkl"
}

models = {}
for name, path in model_files.items():
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        elif path.endswith(".keras") or path.endswith(".h5"):
            models[name] = load_model(path)
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
        return "❌ Missing input(s)", "Please make sure all fields are filled.", ""

    if sc_x:
        input_array = sc_x.transform(input_array)

    model = models.get(model_choice)
    if not model:
        return "Model not found", "Model not found", ""

    pred_raw = model.predict(input_array)
    prediction = int(pred_raw[0]) if pred_raw.shape == (1,) else int(pred_raw[0][0] >= 0.5) if pred_raw.shape == (1, 1) else int(np.argmax(pred_raw[0]))

    risk = "High Risk" if prediction == 1 else "Low Risk"
    advice = get_dynamic_response_gemini(prediction, inputs)
    filename = f"Heart_Report_{model_choice}.pdf"
    generate_health_report_pdf(inputs, prediction, advice, filename)

    return risk, advice, filename

# === UI Layout ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", font=["Inter", "sans-serif"])) as demo:
    gr.Markdown("""
    <div style='text-align: center; font-size: 28px; font-weight: bold;'>
        ❤️ Heart Disease Risk Predictor + Gemini Assistant
    </div>
    <div style='text-align: center; font-size: 16px; color: gray;'>
        Quickly evaluate heart disease risk and receive personalized AI-generated health guidance.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            age = gr.Number(label="Age", value=50)
            sex = gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)", value=1, interactive=True)
            cp = gr.Dropdown([0, 1, 2, 3], label="Chest Pain Type", value=0)
            trestbps = gr.Number(label="Resting BP (mm Hg)", value=120)
            chol = gr.Number(label="Serum Cholesterol (mg/dl)", value=200)
            fbs = gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl", value=0)
            restecg = gr.Dropdown([0, 1, 2], label="Resting ECG Result", value=0)
            thalach = gr.Number(label="Max Heart Rate", value=150)
            exang = gr.Radio([0, 1], label="Exercise Induced Angina", value=0)
            oldpeak = gr.Number(label="ST Depression", value=1.0)
            slope = gr.Dropdown([0, 1, 2], label="Slope of ST Segment", value=1)
            ca = gr.Dropdown([0, 1, 2, 3], label="Major Vessels Colored", value=0)
            model_choice = gr.Dropdown(list(models.keys()), label="Prediction Model", value="Random Forest")

            submit = gr.Button("🔍 Predict Risk", size="lg")

        with gr.Column(scale=1):
            risk_out = gr.Textbox(label="💡 Risk Assessment", lines=1)
            advice_out = gr.Textbox(label="💬 Gemini AI Health Advice", lines=8)
            pdf_link = gr.File(label="📄 Download Your PDF Report")

    submit.click(
        predict_heart_disease,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, model_choice],
        outputs=[risk_out, advice_out, pdf_link]
    )

if __name__ == "__main__":
    demo.launch()
