import joblib
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from app.preprocess import preprocess_data

def train_model(df: pd.DataFrame, target_col: str, model_path: str):
    """Train an XGBoost model with KFold cross-validation"""
    X, y, encoders = preprocess_data(df, target_col)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    metrics = {
        "roc_auc": roc_auc_score(y_true_all, y_prob_all),
        "report": classification_report(y_true_all, y_pred_all, output_dict=True),
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders}, model_path)

    print(f"✅ Model saved to {model_path}")
    return metrics
