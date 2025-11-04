"""Variable importance training, aggregation and visualization utilities.

Core flow:
 - train_xgb_and_agg_importances(): fits XGBoost on top of a preprocessor
     pipeline and aggregates feature importances back to original raw variables.
 - build_filtered_preprocessor(): rebuilds the ColumnTransformer keeping only
     variables above an importance threshold (honoring priority-binning rules).
 - plot_importances(): convenience barplot of top important variables.
 - dropped_variables_table(): tabular view of variables eliminated by threshold.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from .feature_grouping import parse_df_groups_table
from .preprocessing import QuantileDiscretizer
from feature_engine.imputation import CategoricalImputer, ArbitraryNumberImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_output_to_raw_mapping(preprocessor: ColumnTransformer) -> List[str]:
    """Map each transformed output column back to its originating raw variable.

    Handles OneHotEncoder blocks (binary/cat_low) and TargetEncoder (cat_high),
    numeric pipelines (winsor/bins), and preserves alignment with model inputs.
    """
    mapping: List[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if trans == 'drop' or cols is None:
            continue
        if isinstance(cols, (np.ndarray, pd.Index, tuple)):
            cols = list(cols)
        elif not isinstance(cols, list):
            cols = [cols]
        if len(cols) == 0:
            continue
        if isinstance(trans, Pipeline):
            if name in ('cat_low', 'binary') and 'onehot' in trans.named_steps:
                oh = trans.named_steps['onehot']
                if hasattr(oh, 'categories_'):
                    for i, col in enumerate(cols):
                        n_cats = len(oh.categories_[i])
                        drop = 0
                        if oh.drop == 'first':
                            drop = 1
                        elif oh.drop == 'if_binary' and n_cats == 2:
                            drop = 1
                        n_out = max(0, n_cats - drop)
                        mapping.extend([col] * n_out)
                    continue
            if name == 'cat_high':  # target encoder
                mapping.extend(cols)
                continue
            if name in ('num_winsor', 'num_bins'):
                mapping.extend(cols)
                continue
        mapping.extend(cols)
    return mapping


def train_xgb_and_agg_importances(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    scale_pos_weight: float = 2.8,
) -> Tuple[pd.Series, Pipeline]:
    """Fit XGB on top of `preprocessor` and aggregate importances to raw vars.

    Returns
    -------
    (agg_importances, pipeline_xgb)
        agg_importances: pd.Series sorted desc with raw_var -> importance
        pipeline_xgb: fitted Pipeline(preprocessor, XGBClassifier)
    """
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(random_state=random_state, scale_pos_weight=scale_pos_weight))
    ])
    pipeline_xgb.fit(X_train, y_train)

    preproc_fitted = pipeline_xgb.named_steps['preprocessor']
    model_fitted = pipeline_xgb.named_steps['model']

    importances_arr = model_fitted.feature_importances_
    raw_mapping = build_output_to_raw_mapping(preproc_fitted)

    n = min(len(importances_arr), len(raw_mapping))
    importances_arr = importances_arr[:n]
    raw_mapping = raw_mapping[:n]

    imp_df = pd.DataFrame({'raw_var': raw_mapping, 'importance': importances_arr})
    agg_importances = imp_df.groupby('raw_var', as_index=True)['importance'].sum().sort_values(ascending=False)
    return agg_importances, pipeline_xgb


def build_filtered_preprocessor(
    df_groups_final: pd.DataFrame,
    df_model: pd.DataFrame,
    important_vars: List[str]
) -> ColumnTransformer:
    """Rebuild preprocessor keeping only variables present in `important_vars`.

    Preserves priority binning rules for AGE and MONTHS_IN_THE_JOB.
    """
    _group_vars_all = parse_df_groups_table(df_groups_final)

    binary_flags_f = [c for c in _group_vars_all.get('binary', []) if c in important_vars]
    num_reales_f   = [c for c in _group_vars_all.get('num_reales', []) if c in important_vars]
    cat_low_f      = [c for c in _group_vars_all.get('cat_low', []) if c in important_vars]
    cat_high_f     = [c for c in _group_vars_all.get('cat_high', []) if c in important_vars]

    priority_candidates = [
        c for c in ['AGE', 'MONTHS_IN_THE_JOB']
        if c in df_model.columns and np.issubdtype(df_model[c].dtype, np.number)
    ]

    num_bins_f: List[str] = []
    num_winsor_f: List[str] = list(num_reales_f)

    for p in priority_candidates:
        if p in cat_low_f:
            cat_low_f.remove(p)
        if p in num_winsor_f:
            num_winsor_f.remove(p)
        if p in important_vars and p not in num_bins_f:
            num_bins_f.append(p)

    # Dedup
    num_winsor_f = list(dict.fromkeys(num_winsor_f))
    num_bins_f   = list(dict.fromkeys(num_bins_f))
    binary_flags_f = list(dict.fromkeys(binary_flags_f))
    cat_low_f    = list(dict.fromkeys(cat_low_f))
    cat_high_f   = list(dict.fromkeys(cat_high_f))

    num_winsor_pipeline_f = Pipeline(steps=[
        ('imputer', ArbitraryNumberImputer(arbitrary_number=-1)),
        ('winsor', Winsorizer(capping_method='quantiles', tail='both', fold=0.01)),
        ('scaler', RobustScaler())
    ])

    num_bins_pipeline_f = Pipeline(steps=[
        ('imputer', ArbitraryNumberImputer(arbitrary_number=-1)),
        ('qbins', QuantileDiscretizer(n_bins=4))
    ])

    binary_pipeline_f = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
    ])

    cat_low_pipeline_f = Pipeline(steps=[
        ('imputer', CategoricalImputer(imputation_method='missing', fill_value='Not Informed')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_high_pipeline_f = Pipeline(steps=[
        ('imputer', CategoricalImputer(imputation_method='missing', fill_value='Not Informed')),
        ('target_enc', __import__('category_encoders').TargetEncoder(smoothing=0.3))
    ])

    preprocessor_filtered = ColumnTransformer(
        transformers=[
            ('num_winsor', num_winsor_pipeline_f, num_winsor_f),
            ('num_bins',   num_bins_pipeline_f,   num_bins_f),
            ('binary',     binary_pipeline_f,     binary_flags_f),
            ('cat_low',    cat_low_pipeline_f,    cat_low_f),
            ('cat_high',   cat_high_pipeline_f,   cat_high_f),
        ],
        remainder='drop'
    )
    return preprocessor_filtered


def plot_importances(agg_importances: pd.Series, threshold: float, top_k: int = 20) -> None:
    """Plot top_k variables with importance >= threshold using seaborn barplot."""
    agg_imp_filtered = agg_importances[agg_importances >= threshold]
    K_plot = min(top_k, len(agg_imp_filtered))
    plt.figure(figsize=(8, 6))
    sns.barplot(x=agg_imp_filtered.iloc[:K_plot], y=agg_imp_filtered.index[:K_plot], orient='h')
    plt.title("Variables Importantes (>= umbral), nombres originales")
    plt.xlabel("Importancia (agregada)")
    plt.ylabel("Variable original")
    plt.tight_layout()
    plt.show()


def dropped_variables_table(agg_importances: pd.Series, threshold: float) -> pd.DataFrame:
    """Return a table of variables eliminated by the given importance threshold."""
    dropped_imp = agg_importances[agg_importances < threshold]
    if len(dropped_imp) == 0:
        return pd.DataFrame(columns=['variable', 'importance', 'threshold'])
    dropped_df = dropped_imp.reset_index()
    dropped_df.columns = ['variable', 'importance']
    dropped_df['threshold'] = threshold
    return dropped_df.sort_values('importance', ascending=False)
