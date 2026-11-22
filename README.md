# Credit Card Default Prediction

Predict whether a credit card customer will **default (1)** or **not default (0)**, based on demographic, financial and credit-history data from the American Express dataset. The project includes the exploratory notebook and a **Streamlit web app** where you upload a CSV and download predictions. 
App url: https://credit-risk-defaultpredictor.streamlit.app/

## Project overview

| Item | Details |
|---|---|
| Problem | Binary classification: `credit_card_default` (0 = no default, 1 = default) |
| Data | `American_Express_dataset.csv` (~45.5k customers, 18 features, ~8% defaults) |
| Models | Logistic Regression (used in the app) and Random Forest (compared in the notebook) |
| Test performance (Logistic Regression) | Accuracy ≈ 0.98, default-class precision 0.98, recall 0.77 |

### Preprocessing
- Dropped identifiers (`customer_id`, `name`) and rows with missing numeric values
- Removed `Unknown` occupations and the single `XNA` gender row; down-sampled the over-represented `Laborers` group
- One-hot encoded `gender`, `owns_car`, `owns_house`, `migrant_worker`, `occupation_type`
- Min-max scaled numeric features; stratified 70/30 train/test split

### Input columns required by the app
`gender, owns_car, owns_house, migrant_worker, occupation_type, age, no_of_children, net_yearly_income, no_of_days_employed, total_family_members, yearly_debt_payments, credit_limit, credit_limit_used(%), credit_score, prev_defaults, default_in_last_6months`

Extra columns (e.g. `customer_id`, `name`) are allowed and passed through to the output.

## Files

| File | Purpose |
|---|---|
| `Credit_risk_prediction_STB.ipynb` | EDA, preprocessing, model training and evaluation |
| `train_model.py` | Reproduces the preprocessing + logistic regression as one pipeline and saves `model.joblib` |
| `app.py` | Streamlit app |
| `model.joblib` | Trained pipeline (preprocessing + model) |
| `sample_input.csv` | Example upload file |
| `American_Express_dataset.csv` | Training data |

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with the required columns (try `sample_input.csv`). The app shows the predicted `credit_card_default` (0/1) and the default probability per row, summary metrics, and a button to download the results.

To retrain the model: `python train_model.py`.

## License
See [LICENSE](LICENSE).
