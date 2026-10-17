"""Leakage-safe preprocessing and model pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# In the common UCI Cleveland dataset, these variables are coded categories.
CATEGORICAL_CANDIDATES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def make_preprocessor(X):
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

def make_models(X, random_state=42):
    return {
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=random_state, n_jobs=-1
            )),
        ]),
    }
