import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

print("Training dummy model...")

np.random.seed(42)
data = pd.DataFrame({
    "age": np.random.randint(20, 70, 1000),
    "income": np.random.randint(10000, 80000, 1000),
    "loan_amount": np.random.randint(1000, 20000, 1000),
})
data["default"] = ((data["loan_amount"] / data["income"]) * data["age"] > 0.6).astype(int)

X = data[["age", "income", "loan_amount"]]
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Model trained. ROC-AUC: {roc:.4f}")

os.makedirs("app/model", exist_ok=True)
joblib.dump(model, "app/model/model.pkl")
print("Model saved to app/model/model.pkl")
