# 📊 Customer Churn Prediction

A Streamlit web app that predicts how likely a telecom customer is to leave (churn) and shows the top factors behind each prediction.

## Features

- **Dashboard**: churn rate, churn distribution, contract type and monthly charges charts
- **Dataset Explorer**: preview, statistics and missing values
- **Model Training**: Logistic Regression pipeline with a confusion matrix and classification report
- **Churn Prediction**: enter a customer's details to get the churn probability
- **Top Contributing Factors**: shows which features increase or decrease that customer's churn risk

## Tech Stack

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, from Kaggle). Place `WA_Fn-UseC_-Telco-customer-Churn.csv` in the same folder as `app.py`.

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install streamlit pandas numpy scikit-learn plotly
streamlit run app.py
```

## How It Works

1. Missing values are imputed, numeric features are scaled and categorical features are one-hot encoded.
2. A Logistic Regression model (`class_weight="balanced"`) is trained on an 80/20 split.
3. For each prediction, the model's coefficients are used to find which features push the churn risk up or down.

> The factors reflect patterns learned by the model, not guaranteed causes of churn.

## Usage

1. Open **Train Model** and click **Train Model**.
2. Go to **Predict Churn**, enter customer details and click **Predict Churn**.

## Author

**Your Name** - [GitHub](https://github.com/Hajra-Javed-Awan)

