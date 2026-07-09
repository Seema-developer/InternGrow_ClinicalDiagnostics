import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Clinical Disease Diagnostics Engine",
    page_icon="🩺",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("model_name.pkl", "rb") as f:
        model_name = pickle.load(f)
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)
    return model, scaler, feature_names, model_name, results

try:
    model, scaler, feature_names, model_name, results = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

FEATURE_RANGES = {
    "mean radius":       (6.0, 30.0, 14.0, 0.1),
    "mean texture":       (9.0, 40.0, 19.0, 0.1),
    "mean perimeter":     (40.0, 190.0, 92.0, 1.0),
    "mean area":          (140.0, 2500.0, 655.0, 5.0),
    "mean smoothness":    (0.05, 0.17, 0.096, 0.001),
    "mean compactness":   (0.02, 0.35, 0.104, 0.001),
    "mean concavity":     (0.0, 0.43, 0.089, 0.001),
    "mean symmetry":      (0.10, 0.30, 0.181, 0.001),
    "worst radius":       (7.0, 37.0, 16.3, 0.1),
    "worst concavity":    (0.0, 1.25, 0.272, 0.001),
}

st.sidebar.header("🩺 Patient Metrics")
patient_inputs = {}
for feat in feature_names:
    lo, hi, default, step = FEATURE_RANGES[feat]
    patient_inputs[feat] = st.sidebar.slider(
        feat.title(), min_value=lo, max_value=hi, value=default, step=step
    )

st.title("🩺 Clinical Disease Diagnostics Engine")
st.write("DEBUG MODE - testing without SHAP")

col1, col2, col3 = st.columns(3)
col1.metric("Model", model_name)
col2.metric("Test ROC-AUC", f"{results[model_name]['roc_auc']:.3f}")
col3.metric("Test Accuracy", f"{results[model_name]['accuracy']:.3f}")

st.divider()

input_df = pd.DataFrame([patient_inputs])[feature_names]
input_scaled = scaler.transform(input_df)

proba = model.predict_proba(input_scaled)[0]
malignant_prob = proba[0]
benign_prob = proba[1]
prediction = "Benign" if benign_prob > malignant_prob else "Malignant"

st.subheader("Real-Time Risk Assessment")
if prediction == "Malignant":
    st.error(f"⚠️ Predicted: **{prediction}**")
else:
    st.success(f"✅ Predicted: **{prediction}**")

prob_df = pd.DataFrame({
    "Outcome": ["Malignant", "Benign"],
    "Probability": [malignant_prob, benign_prob],
})
st.bar_chart(prob_df.set_index("Outcome"))