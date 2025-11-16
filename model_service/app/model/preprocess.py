import os
import pandas as pd

BASE_DIR = "/app/artifacts"
VARS_XLS_PATH = os.path.join(BASE_DIR, "PAKDD2010_VariablesList.XLS")

print("🔹 Loading expected column names...")

df_vars = pd.read_excel(VARS_XLS_PATH)
colnames = df_vars["Var_Title"].astype(str).tolist()

# Fix XLS bug
if len(colnames) > 43:
    colnames[43] = "MATE_" + colnames[43]

# Remove target-like columns
EXPECTED_COLS = [
    c for c in colnames
    if c.upper() not in ["TARGET_LABEL_BAD=1", "TARGET"]
    and not c.upper().startswith("ID_")
]

print(f"Loaded {len(EXPECTED_COLS)} expected columns")
