import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


bundle = load_model()
pipeline, cat_cols, num_cols = bundle["pipeline"], bundle["cat_cols"], bundle["num_cols"]
required = cat_cols + num_cols

st.title("💳 Credit Card Default Predictor")
st.write(
    "Upload a CSV with customer data and get a **credit_card_default** prediction "
    "(`0` = no default, `1` = default) for every row, from a trained logistic regression model."
)

with st.expander("Required columns"):
    st.code(", ".join(required))
    st.caption(
        "Extra columns (e.g. customer_id, name, credit_card_default) are kept in the output "
        "but ignored by the model."
    )
    try:
        with open("sample_input.csv", "rb") as f:
            st.download_button("Download sample CSV", f, "sample_input.csv", "text/csv")
    except FileNotFoundError:
        pass

uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    X = df[required].copy()
    for c in num_cols + ["migrant_worker"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    out = df.copy()
    out["predicted_credit_card_default"] = pipeline.predict(X)
    out["default_probability"] = pipeline.predict_proba(X)[:, 1].round(4)

    n_def = int(out["predicted_credit_card_default"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(out))
    c2.metric("Predicted defaults", n_def)
    c3.metric("Default rate", f"{n_def / len(out):.1%}")

    st.dataframe(out, use_container_width=True)
    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
    )
