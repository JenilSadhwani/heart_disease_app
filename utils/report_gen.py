import google.generativeai as genai
from fpdf import FPDF
import qrcode
from datetime import datetime
import re

genai.configure(api_key="AIzaSyA51GP23aYcWQgmtmNfNqTqkHzyMfHnj5k")
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

def clean_text_for_pdf(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_dynamic_response_gemini(prediction, inputs):
    risk_info = "The patient is at high risk for heart disease." if prediction == 1 else "Low risk."
    prompt = (
        f"{risk_info}\n\nPatient details:\n" +
        "\n".join(f"- {k}: {v}" for k, v in inputs.items()) +
        "\n\nProvide health tips and lifestyle guidance."
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

def generate_health_report_pdf(patient_data, prediction, gemini_advice, filename="report.pdf", qr_data=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Heart Disease Risk Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Data:", ln=True)
    pdf.set_font("Arial", size=11)
    for key, val in patient_data.items():
        pdf.cell(0, 8, f"{key}: {val}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Prediction:", ln=True)
    pdf.set_font("Arial", size=11)
    result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk"
    pdf.multi_cell(0, 8, result_text)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gemini AI Advice:", ln=True)
    pdf.set_font("Arial", size=11)
    cleaned_advice = clean_text_for_pdf(gemini_advice)
    for line in cleaned_advice.split('\n'):
        pdf.multi_cell(0, 8, line)

    if qr_data:
        qr = qrcode.make(qr_data)
        qr.save("qr_temp.png")
        pdf.image("qr_temp.png", x=170, y=10, w=30)

    pdf.output(filename)
