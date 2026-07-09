# InternGrow ML Track - Task 4: Clinical Disease Diagnostics Engine

Predicts tumor malignancy risk from biopsy measurements using supervised
classification models, with an interactive Streamlit dashboard for
real-time risk assessment.

## What this does (Task requirements covered)

**Base task:** Apply supervised classification models (SVM, Random Forest,
XGBoost) on a structured medical dataset to evaluate health risk profiles.
- Uses the Breast Cancer Wisconsin (Diagnostic) dataset — a real clinical
  dataset (569 patients, biopsy measurements, malignant/benign labels).
- Trains and compares all three required models: SVM, Random Forest, XGBoost.
- Reports Accuracy, F1-score, and ROC-AUC for each, and auto-selects the
  best performer.

**Upgrade feature:** Interactive Streamlit web dashboard where a "doctor"
inputs patient metrics via sliders and sees real-time disease probability
charts.
- 10 clinically-relevant sliders (radius, texture, perimeter, area,
  smoothness, compactness, concavity, symmetry, worst-radius, worst-concavity)
- Real-time bar chart of malignant vs. benign probability
- SHAP waterfall plot showing *why* the model made that prediction
  (explainable AI — which features pushed the risk up or down)

## How to run it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (creates model.pkl, scaler.pkl, etc.)
python train_model.py

# 3. Launch the dashboard
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Results (from a sample training run)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| SVM | 0.947 | 0.959 | 0.987 |
| Random Forest | 0.939 | 0.951 | **0.992** |
| XGBoost | 0.956 | 0.966 | 0.990 |

Random Forest was auto-selected as the best model by ROC-AUC (the standard
metric for medical risk-scoring models, since it accounts for the full
probability threshold range rather than just one cutoff).

## Project structure
