"""Train the logistic regression credit-default model and save it to model.joblib.

Mirrors the preprocessing in Credit_risk_prediction_STB.ipynb, but wraps it in a single
sklearn Pipeline so the Streamlit app applies exactly the same transformations.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

DATA_PATH = "American_Express_dataset.csv"
TARGET = "credit_card_default"
CAT_COLS = ["gender", "owns_car", "owns_house", "migrant_worker", "occupation_type"]
NUM_COLS = [
    "age", "no_of_children", "net_yearly_income", "no_of_days_employed",
    "total_family_members", "yearly_debt_payments", "credit_limit",
    "credit_limit_used(%)", "credit_score", "prev_defaults", "default_in_last_6months",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop(columns=["customer_id", "name"])
    df = df.dropna(subset=[c for c in df.columns if c not in CAT_COLS] )
    df = df[df["occupation_type"] != "Unknown"]
    df = df[df["gender"] != "XNA"]
    # Same down-sampling of the over-represented 'Laborers' class as in the notebook
    drop_idx = df[df["occupation_type"] == "Laborers"].sample(n=192, random_state=42).index
    return df.drop(drop_idx)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), CAT_COLS),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ]), NUM_COLS),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    X, y = df[CAT_COLS + NUM_COLS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    pipe = build_pipeline().fit(X_train, y_train)
    print(classification_report(y_test, pipe.predict(X_test)))
    joblib.dump({"pipeline": pipe, "cat_cols": CAT_COLS, "num_cols": NUM_COLS}, "model.joblib")
    print("Saved model.joblib")
