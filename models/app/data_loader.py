import pandas as pd
import os

def load_credit_dataset(raw_dir="data/raw"):
    xls_path = os.path.join(raw_dir, "PAKDD2010_VariablesList.XLS")
    txt_path = os.path.join(raw_dir, "PAKDD2010_Modeling_Data.txt")

    if not os.path.exists(xls_path) or not os.path.exists(txt_path):
        raise FileNotFoundError("Cannot find the required dataset files in the specified directory.")

    variables_df = pd.read_excel(xls_path)
    colnames = variables_df["Var_Title"].astype(str).tolist()

    # Add MATE_ prefix to column 44 (index 43)
    if len(colnames) > 43:
        colnames[43] = "MATE_" + colnames[43]

    df = pd.read_csv(txt_path, sep="\t", encoding="latin1", low_memory=False, header=None, names=colnames)
    print(f"✅ Data loaded: {df.shape}")
    return df
