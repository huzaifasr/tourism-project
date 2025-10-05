# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
import numpy as np

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/huzaifa-sr/tourism-project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop columns not required
first_col = df.columns[0]
df = df.drop(columns=[first_col])
print(f"Dropped unnamed index column: {first_col}")
df = df.drop(columns=["CustomerID"])
print("Dropped CustomerID column")


# Fix Gender typos/variants seen in CSV (e.g. 'Fe Male')
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    df.loc[df["Gender"].str.contains(r"fe|fem", na=False), "Gender"] = "Female"
    df.loc[df["Gender"].str.contains(r"male", na=False) & ~df["Gender"].str.contains(r"fe|fem", na=False), "Gender"] = "Male"
    df.loc[~df["Gender"].isin(["Male", "Female"]) , "Gender"] = np.nan


# Columns to treat as categorical for encoding
categorical_cols = [
    c for c in [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
        "CityTier",
        "PreferredPropertyStar",
        "Passport",
        "PitchSatisfactionScore",
        "ProductPitched"
    ]
    if c in df.columns
]

# Numeric columns detection (excluding target)
numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET]]

# Convert numeric-like columns to numeric dtype where possible
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Impute missing values: numeric -> median, categorical -> mode
for c in numeric_cols:
    if df[c].isna().any():
        med = df[c].median()
        df[c] = df[c].fillna(med)
        print(f"Imputed numeric column {c} with median={med}")

for c in categorical_cols:
    if df[c].isna().any():
        mode_val = df[c].mode(dropna=True)
        if not mode_val.empty:
            mode_val = mode_val[0]
            df[c] = df[c].fillna(mode_val)
            print(f"Imputed categorical column {c} with mode='{mode_val}'")
        else:
            # if mode cannot be determined, fill with string 'Unknown'
            df[c] = df[c].fillna("Unknown")
            print(f"Filled categorical column {c} with 'Unknown' (no mode available)")

# One-hot encode categorical columns using pandas get_dummies (drop first to avoid collinearity)
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, prefix_sep="__", drop_first=True)
    print(f"One-hot encoded columns: {categorical_cols}")

# Final check: ensure no missing values remain
missing_after = df.isna().sum().sum()
print(f"Total missing values after imputation/encoding: {missing_after}")

# Split into X and y
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Ensure output directory exists
out_dir = "tourism_project/data/prepared"
os.makedirs(out_dir, exist_ok=True)

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

# Save CSVs
Xtrain.to_csv(os.path.join(out_dir, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(out_dir, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(out_dir, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(out_dir, "ytest.csv"), index=False)

print(f"Saved prepared files to {out_dir}")

# Optional: upload to Hugging Face dataset repo if HF_TOKEN present in env
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    api = HfApi(token=hf_token)
    repo_id = "huzaifa-sr/tourism-project"
    try:
        for filename in ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]:
            path = os.path.join(out_dir, filename)
            api.upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=repo_id, repo_type="dataset")
            print(f"Uploaded {filename} to Hugging Face dataset {repo_id}")
    except Exception as e:
        print(f"Failed to upload to HF: {e}")

print("Data preparation completed.")
