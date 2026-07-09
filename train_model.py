import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import pickle

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("InternGrow Task 4 - Clinical Disease Diagnostics Engine")
    print("=" * 60)

    # 1. Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign
    print(f"\nDataset: {X.shape[0]} patients, {X.shape[1]} clinical features")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))} "
          f"(0=malignant, 1=benign)")

    # 2. Pick top clinically-intuitive features for the dashboard sliders
    #    (keeps the UI usable instead of showing all 30 raw features)
    top_features = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean symmetry", "worst radius", "worst concavity"
    ]
    X_top = X[top_features]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4. Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train and compare 3 models required by the task
    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
    }

    results = {}
    trained_models = {}
    print("\n--- Model Comparison ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        results[name] = {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}
        trained_models[name] = model
        print(f"{name:15s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # 6. Pick the best model by ROC-AUC (standard for medical risk models)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
    print("\nClassification report for best model:")
    print(classification_report(y_test, best_model.predict(X_test_scaled),
                                 target_names=["Malignant", "Benign"]))

    # 7. Save everything the Streamlit app needs
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(top_features, f)
    with open("model_name.pkl", "wb") as f:
        pickle.dump(best_name, f)
    # Save a background sample for SHAP explainer (needed at app runtime)
    with open("background_data.pkl", "wb") as f:
        pickle.dump(X_train_scaled[:100], f)
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nSaved: model.pkl, scaler.pkl, feature_names.pkl, model_name.pkl, "
          "background_data.pkl, results.pkl")
    print("Now run: streamlit run app.py")


if __name__ == "__main__":
    main()