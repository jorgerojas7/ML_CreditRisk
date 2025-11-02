"""Utilities to compute missingness and group variables for preprocessing.

Key concepts
------------
- Groups: 'binary', 'num_reales', 'cat_low', 'cat_high' and a special
    'Un solo valor' for constant columns.
- make_rectangular(): creates a display-friendly DataFrame summarizing groups
    with Missing% per variable for auditing and downstream parsing.
"""

from typing import Dict, List, Tuple
import pandas as pd


def compute_missing_pct(df: pd.DataFrame, treat_empty_as_missing: bool = True) -> Dict[str, float]:
    """Compute percentage of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    treat_empty_as_missing : bool, default True
        For object columns, count empty strings '' (after strip) as missing.

    Returns
    -------
    Dict[str, float]
        Mapping column_name -> missing_percentage (0-100).
    """
    if treat_empty_as_missing:
        obj_cols = df.select_dtypes(include=['object']).columns
        empty_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        if len(obj_cols) > 0:
            empty_mask[obj_cols] = df[obj_cols].astype(str).apply(lambda s: s.str.strip() == '')
        missing_mask = df.isna() | empty_mask
    else:
        missing_mask = df.isna()
    return (missing_mask.mean() * 100).to_dict()


def build_groups_raw(
    df_model: pd.DataFrame,
    target_col: str,
    single_value_cols: List[str],
    low_card_thres: int,
    num_real_unique_min: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build base groups from df_model and return (groups_raw, groups_lists).

    Rules
    -----
    - binary: <=2 unique non-null values
    - num_reales: numeric with unique_count >= num_real_unique_min
    - cat_low: the rest of numeric discrete and object with <= low_card_thres
    - cat_high: object with > low_card_thres
    - 'Un solo valor': provided single_value_cols
    """
    assigned = {target_col}
    binary_flags: List[str] = []
    num_reales: List[str] = []
    cat_low: List[str] = []
    cat_high: List[str] = []

    # Binary (any dtype with <=2 unique non-null values)
    for col in df_model.columns:
        if col in assigned or col == target_col:
            continue
        if df_model[col].dropna().nunique() <= 2:
            binary_flags.append(col)
            assigned.add(col)

    # Numeric real vs discrete by unique count
    for col in df_model.select_dtypes(include=['number']).columns:
        if col in assigned or col == target_col:
            continue
        if df_model[col].nunique() >= num_real_unique_min:
            num_reales.append(col)
            assigned.add(col)
        else:
            cat_low.append(col)
            assigned.add(col)

    # Categoricals (object): low/high cardinality
    for col in df_model.select_dtypes(include=['object']).columns:
        if col in assigned or col == target_col:
            continue
        (cat_low if df_model[col].nunique() <= low_card_thres else cat_high).append(col)
        assigned.add(col)

    groups_raw = {
        'Un solo valor': single_value_cols,
        'binary': binary_flags,
        'num_reales': num_reales,
        'cat_low': cat_low,
        'cat_high': cat_high,
    }
    groups_lists = {
        'binary': binary_flags,
        'num_reales': num_reales,
        'cat_low': cat_low,
        'cat_high': cat_high,
    }
    return groups_raw, groups_lists


def make_rectangular(groups: Dict[str, List[str]], missing_pct: Dict[str, float]) -> pd.DataFrame:
    """Return a rectangular table of groups with Missing% for display/audit.

    Each cell contains "<var> | MV xx.xx%" or None for padding.
    """
    disp = {g: [f"{c} | MV {missing_pct.get(c, 0):.2f}%" for c in cols] for g, cols in groups.items()}
    max_len = max((len(v) for v in disp.values()), default=0)
    for k in disp:
        disp[k] = list(disp[k]) + [None] * (max_len - len(disp[k]))
    return pd.DataFrame(disp)


def parse_df_groups_table(df_table: pd.DataFrame) -> dict:
    """Parse rectangular df_groups_final into {group: [variables]}.

    The variable name is extracted as the substring before the first '|' in
    each non-null cell.
    """
    result = {}
    for g in df_table.columns:
        vals = [v for v in df_table[g].dropna().tolist() if isinstance(v, str)]
        vars_ = []
        for val in vals:
            name = val.split('|', 1)[0].strip()
            if name:
                vars_.append(name)
        result[g] = vars_
    return result
