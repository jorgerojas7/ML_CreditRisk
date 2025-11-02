"""Model definitions, evaluators, and specialized pipelines for credit risk.

This module exposes:
 - get_base_models(): a dictionary of baseline estimators (RF, XGBoost, and
     optionally LightGBM/CatBoost if installed).
 - get_xgb_leaves_lr(): a GB-leaves -> OneHot -> LogisticRegression pipeline.
 - evaluate_models(): utility to train/evaluate models with a shared preprocessor.

All models are intended to be used inside a sklearn Pipeline with a
ColumnTransformer preprocessor that handles imputation/encoding.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


def get_base_models() -> Dict[str, object]:
    """Return a dict of baseline classifiers to compare.

    Keys are human-friendly model names; values are sklearn-compatible
    estimators. If LightGBM or CatBoost are not installed, those entries
    are skipped silently.
    """
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=10000, tol=1e-3, class_weight='balanced', solver='saga'),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features=0.5,
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.8,
            random_state=42,
            eval_metric='auc'
        ),
    }

    # Optional: LightGBM (if installed)
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            metric='auc'
        )
    except Exception:
        # Silently skip if package not available
        pass

    # Optional: CatBoost (if installed)
    try:
        from catboost import CatBoostClassifier  # type: ignore
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            subsample=0.8,
            random_seed=42,
            eval_metric='AUC',
            loss_function='Logloss',
            scale_pos_weight=3.8,
            verbose=0
        )
    except Exception:
        # Silently skip if package not available
        pass

    return models


def get_xgb_lr_stacking() -> Tuple[str, object]:
    """Return a (name, model) tuple for a stacking model: XGBoost as base, LogisticRegression as final estimator.

    This effectively learns a logistic calibration/meta-model over XGBoost's probabilistic outputs using cross-validated
    out-of-fold predictions (stack_method='predict_proba').
    """
    base_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.8,
        random_state=42,
        eval_metric='auc'
    )
    final_lr = LogisticRegression(max_iter=10000, tol=1e-3, class_weight='balanced', solver='saga')
    stacking = StackingClassifier(
        estimators=[('xgb', base_xgb)],
        final_estimator=final_lr,
        stack_method='predict_proba',
        passthrough=False,
        cv=5
    )
    return "XGB+LR (stacking)", stacking


class LeafIndexEncoder(BaseEstimator, TransformerMixin):
    """Fit an XGBClassifier and transform samples to leaf indices per tree.

    Output shape: (n_samples, n_trees). The indices are integers suitable for
    subsequent one-hot encoding to capture tree-structure interactions.
    """
    def __init__(self,
                 n_estimators: int = 200,
                 learning_rate: float = 0.05,
                 max_depth: int = 4,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 scale_pos_weight: float = 3.8,
                 random_state: int = 42,
                 eval_metric: str = 'auc'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.model_ = None

    def fit(self, X, y):
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
        )
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        # XGBClassifier.apply returns leaf indices per tree (float); cast to int
        leaf_idx = self.model_.apply(X)
        # Some versions return shape (n_samples, n_trees, 1); squeeze if needed
        leaf_idx = np.array(leaf_idx)
        if leaf_idx.ndim == 3:
            leaf_idx = leaf_idx[:, :, 0]
        return leaf_idx.astype(int)


def get_xgb_leaves_lr() -> Tuple[str, object]:
    """Return (name, estimator) for the GB-leaves -> OneHot -> LogisticRegression pipeline.

    This pipeline first maps each sample to the index of the terminal leaf in
    every XGBoost tree (via LeafIndexEncoder), then applies one-hot encoding to
    the leaf indices, and finally fits a Logistic Regression classifier.
    """
    leaf_encoder = LeafIndexEncoder(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.8,
        random_state=42,
        eval_metric='auc'
    )
    onehot = OneHotEncoder(handle_unknown='ignore')
    lr = LogisticRegression(max_iter=10000, tol=1e-3, class_weight='balanced', solver='saga')
    pipeline = Pipeline(steps=[
        ('leaf_encoder', leaf_encoder),
        ('onehot', onehot),
        ('lr', lr)
    ])
    return "XGB leaves + LR", pipeline


def evaluate_models(models: Dict[str, object], preprocessor, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train/evaluate a dict of estimators with a shared preprocessor.

    Parameters
    ----------
    models : dict
        {model_name: estimator} estimators implementing predict_proba.
    preprocessor : ColumnTransformer
        Fitted or unfitted preprocessor. It will be fit inside a Pipeline.
    X_train, X_test : pd.DataFrame
        Train/test splits with raw features.
    y_train, y_test : pd.Series
        Binary target splits (1 = Bad).

    Returns
    -------
    pd.DataFrame
        Sorted by ROC_AUC descending with columns: Modelo, ROC_AUC, KS,
        Recall (Bad=1), Precision.
    """
    results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_prob)
        recall_bad = recall_score(y_test, y_pred)
        precision_bad = precision_score(y_test, y_pred)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        ks = float(np.max(tpr - fpr))

        results.append({
            "Modelo": name,
            "ROC_AUC": roc_auc,
            "KS": ks,
            "Recall (Bad=1)": recall_bad,
            "Precision": precision_bad
        })
    return pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
