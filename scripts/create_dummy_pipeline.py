import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

if __name__ == "__main__":
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)

    # Minimal training data
    X = pd.DataFrame({'x': [0, 1]})
    y = [0, 1]

    pipe = Pipeline(steps=[('model', DummyClassifier(strategy='uniform', random_state=42))])
    pipe.fit(X, y)

    out_path = models_dir / 'pipeline.joblib'
    joblib.dump(pipe, out_path)
    print(f"✅ Saved pipeline at {out_path}")
