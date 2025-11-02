"""Preprocessing primitives and ColumnTransformer builder for the project.

This module provides:
- QuantileDiscretizer: robust, duplicates-safe discretizer for numeric features.
- build_preprocessor_from_groups(): builds a ColumnTransformer using the
    rectangular groups table, with priority binning for AGE and MONTHS_IN_THE_JOB.
- resumen_columnas(): prints a compact summary of the generated feature space.
"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from feature_engine.imputation import CategoricalImputer, ArbitraryNumberImputer
from feature_engine.outliers import Winsorizer
from sklearn.base import BaseEstimator, TransformerMixin
from .feature_grouping import parse_df_groups_table


class QuantileDiscretizer(BaseEstimator, TransformerMixin):
    """Robust discretizer using quantiles, safe against duplicate edges.

    Parameters
    ----------
    n_bins : int, default 4
        Target number of quantile-based bins. The transformer adapts to the
        number of unique values to avoid degenerate edges.
    """
    def __init__(self, n_bins: int = 4):
        self.n_bins = int(n_bins)
        self.bin_edges_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.columns_ = list(X_df.columns)
        for col in self.columns_:
            s = pd.to_numeric(X_df[col], errors='coerce')
            uniq = s.dropna().nunique()
            if uniq <= 1:
                self.bin_edges_[col] = np.array([s.min() - 1 if pd.notna(s.min()) else -1, s.min() + 1 if pd.notna(s.min()) else 1])
                continue
            q = min(self.n_bins, max(2, uniq))
            try:
                _, bins = pd.qcut(s, q=q, retbins=True, duplicates='drop')
                bins = np.unique(bins)
                if len(bins) < 2:
                    vmin, vmax = s.min(), s.max()
                    bins = np.linspace(vmin, vmax, num=min(self.n_bins, uniq) + 1)
            except Exception:
                vmin, vmax = s.min(), s.max()
                if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                    bins = np.array([0.0, 1.0])
                else:
                    bins = np.linspace(vmin, vmax, num=min(self.n_bins, uniq) + 1)
            self.bin_edges_[col] = bins
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        Xt = np.zeros((len(X_df), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            s = pd.to_numeric(X_df[col], errors='coerce')
            bins = self.bin_edges_[col]
            codes = pd.cut(s, bins=bins, labels=False, include_lowest=True)
            Xt[:, i] = pd.Series(codes).fillna(-1).astype(float).values
        return Xt


def build_preprocessor_from_groups(df_groups_final: pd.DataFrame, df_model: pd.DataFrame) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    """Build ColumnTransformer from the rectangular groups table.

    Gives priority to binning AGE and MONTHS_IN_THE_JOB (if numeric). Returns
    both the preprocessor and a dict with the resolved variable lists per block.
    """
    _group_vars = parse_df_groups_table(df_groups_final)
    binary_flags = _group_vars.get('binary', [])
    num_reales = _group_vars.get('num_reales', [])
    cat_low = _group_vars.get('cat_low', [])
    cat_high = _group_vars.get('cat_high', [])

    # Priority binning for these numeric columns if present
    priority_candidates = [
        c for c in ['AGE', 'MONTHS_IN_THE_JOB']
        if c in df_model.columns and np.issubdtype(df_model[c].dtype, np.number)
    ]

    num_bins: List[str] = []
    num_winsor: List[str] = list(num_reales)
    cat_low = list(cat_low)
    cat_high = list(cat_high)
    binary_flags = list(binary_flags)

    for p in priority_candidates:
        if p in cat_low:
            cat_low.remove(p)
        if p in num_winsor:
            num_winsor.remove(p)
        if p not in num_bins:
            num_bins.append(p)

    # Deduplicate preserving order
    num_winsor = list(dict.fromkeys(num_winsor))
    num_bins = list(dict.fromkeys(num_bins))
    binary_flags = list(dict.fromkeys(binary_flags))
    cat_low = list(dict.fromkeys(cat_low))
    cat_high = list(dict.fromkeys(cat_high))

    # Pipelines
    num_winsor_pipeline = Pipeline(steps=[
        ('imputer', ArbitraryNumberImputer(arbitrary_number=-1)),
        ('winsor', Winsorizer(capping_method='quantiles', tail='both', fold=0.01)),
        ('scaler', RobustScaler())
    ])

    num_bins_pipeline = Pipeline(steps=[
        ('imputer', ArbitraryNumberImputer(arbitrary_number=-1)),
        ('qbins', QuantileDiscretizer(n_bins=4))
    ])

    binary_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    ])

    cat_low_pipeline = Pipeline(steps=[
        ('imputer', CategoricalImputer(imputation_method='missing', fill_value='Not Informed')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_high_pipeline = Pipeline(steps=[
        ('imputer', CategoricalImputer(imputation_method='missing', fill_value='Not Informed')),
        ('target_enc', __import__('category_encoders').TargetEncoder(smoothing=0.3))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_winsor', num_winsor_pipeline, num_winsor),
            ('num_bins', num_bins_pipeline, num_bins),
            ('binary', binary_pipeline, binary_flags),
            ('cat_low', cat_low_pipeline, cat_low),
            ('cat_high', cat_high_pipeline, cat_high),
        ],
        remainder='drop'
    )

    lists = {
        'num_winsor': num_winsor,
        'num_bins': num_bins,
        'binary_flags': binary_flags,
        'cat_low': cat_low,
        'cat_high': cat_high,
    }
    return preprocessor, lists


def resumen_columnas(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series) -> None:
    """Print a summary of generated columns per transformer block.

    Fits the preprocessor on (X, y) and estimates the number of output columns
    produced by each block, attempting to use get_feature_names_out() when
    available or falling back to a small transform() sample.
    """
    print("\n📈 Resumen de columnas generadas por cada bloque:")
    preprocessor.fit(X, y)

    X_sample = X.iloc[: min(1000, len(X))]

    total_cols = 0
    for name, trans, cols in preprocessor.transformers_:
        if trans == 'drop':
            continue
        n_cols = None
        try:
            if hasattr(trans, 'get_feature_names_out'):
                n_cols = len(trans.get_feature_names_out())
            elif isinstance(trans, Pipeline):
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, 'get_feature_names_out'):
                    n_cols = len(last_step.get_feature_names_out())
        except Exception:
            pass
        if n_cols is None:
            try:
                Xt = trans.transform(X_sample[cols])
                n_cols = Xt.shape[1] if getattr(Xt, 'ndim', 2) > 1 else 1
            except Exception:
                n_cols = len(cols)
        total_cols += n_cols
        print(f"  🔹 {name:<12} → {n_cols:>4} columnas (input: {len(cols)})")

    print(f"  ⚙️  Total columnas generadas: {total_cols}\n")
