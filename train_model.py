import pandas as pd
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ================================
# SETTINGS
# ================================
warnings.filterwarnings("ignore")

DATA_PATH = "fraud_detection.csv"   # dataset must be in the same folder
TARGET_COL = "fraud_risk"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# LOAD DATA
# ================================
data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

if TARGET_COL not in data.columns:
    raise ValueError(f"Target column '{TARGET_COL}' missing in dataset")

# Encode all columns
label_encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, os.path.join(MODEL_DIR, "all_label_encoders.joblib"))

# Split features & target
X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# Preprocessing pipeline
pipeline_preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=1,
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("preprocess", pipeline_preprocess),
    ("model", model)
])

# Train and save
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, os.path.join(MODEL_DIR, "XGBoost.joblib"))

print("✅ Model trained and saved successfully!")
