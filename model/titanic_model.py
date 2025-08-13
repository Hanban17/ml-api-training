import seaborn as sns

df = sns.load_dataset("titanic")
import os
import pickle
import sys

import pandas as pd

sys.path.append(os.path.abspath(".."))
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

from model.preprocessing import (
    AgeImputer,
    DropColumnsTransformer,
    EmbarkDeckImputer,
    category_columns,
    numerical_columns,
    ordinal_columns,
)

cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

ord_pipeline = Pipeline(
    [("ordinal", OrdinalEncoder(categories=[[1, 2, 3]]))]  # 1st > 2nd > 3rd
)

num_pipeline = Pipeline([("scaler", StandardScaler())])

preprocessor = Pipeline(
    [
        ("deck_embark_imputer", EmbarkDeckImputer()),
        ("age_imputer", AgeImputer()),
        ("drop_columns", DropColumnsTransformer()),
        (
            "transform",
            ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore"), category_columns),
                    ("ord", OrdinalEncoder(categories=[[1, 2, 3]]), ordinal_columns),
                    ("num", StandardScaler(), numerical_columns),
                ]
            ),
        ),
    ]
)

# === STEP 4: MODEL ===
# Use best tuned params from Optuna
xgb_model = XGBClassifier(
    n_estimators=221,
    max_depth=5,
    learning_rate=0.1022,
    subsample=0.5155,
    colsample_bytree=0.5842,
    gamma=0.1598,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
# === STEP 5: FULL PIPELINE ===
full_pipeline = make_pipeline(preprocessor, xgb_model)

# === STEP 6: TRAIN / TEST ===
X = df.drop(columns="survived")
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

full_pipeline.fit(X_train, y_train)

# === STEP 7: EVALUATE ===
test_acc = full_pipeline.score(X_test, y_test)
print(f"✅ Final Test Accuracy: {test_acc:.4f}")

# === STEP 8: SAVE MODEL ===
joblib.dump(full_pipeline, "titanic_xgboost_pipeline.joblib")
print("💾 Model saved to titanic_xgboost_pipeline.joblib")


with open("titanic_pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)
    pickle.dump(full_pipeline, f)
