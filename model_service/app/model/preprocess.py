import pandas as pd

def preprocess_features(data):
    df = pd.DataFrame(data)

    expected_cols = ["age", "income", "loan_amount"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.fillna(0)
    return df[expected_cols]
