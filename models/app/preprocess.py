import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET_COL = 'TARGET_LABEL_BAD=1'

def preprocess_data(df: pd.DataFrame, target_col=TARGET_COL):
    null_cats = ['MATE_EDUCATION_LEVEL', 'MATE_PROFESSION_CODE', 
                 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'RESIDENCE_TYPE']
    for col in null_cats:
        if col in df.columns:
            df[col].fillna('UNKNOWN', inplace=True)

    if 'MONTHS_IN_RESIDENCE' in df.columns:
        df['MONTHS_IN_RESIDENCE'].fillna(df['MONTHS_IN_RESIDENCE'].median(), inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(0)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
