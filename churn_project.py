
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ==================================================
# 1. FOLDER SETUP
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ==================================================
# 2. FIND AND LOAD KAGGLE DATASET
# ==================================================

def load_dataset():
    csv_files = list(BASE_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in: {BASE_DIR}\n"
            "Please put your Kaggle CSV file in the customer folder."
        )

    for file in csv_files:
        try:
            temp = pd.read_csv(file)
            temp.columns = temp.columns.str.strip()

            if "Churn" in temp.columns:
                print(f"Dataset loaded: {file.name}")
                return temp

        except Exception as error:
            print(f"Could not read {file.name}: {error}")

    raise ValueError(
        "No CSV file with a 'Churn' column was found. "
        "Please check that this is the Telco Customer Churn dataset."
    )


df = load_dataset()

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())


# ==================================================
# 3. DATA CLEANING
# ==================================================

df = df.drop_duplicates().copy()
df = df.dropna(subset=["Churn"]).copy()

df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Convert TotalCharges into numeric values
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Remove ID column if present
df = df.drop(columns=["customerID"], errors="ignore")

if df.empty or df["Churn"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both Churn = Yes and Churn = No."
    )

print("\n========== CLEANED DATA ==========")
print("Shape:", df.shape)
print("\nChurn counts:\n", df["Churn"].value_counts())


# ==================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==================================================

# Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
plt.close()

# Monthly charges
if "MonthlyCharges" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_charges.png", dpi=150)
    plt.close()

# Contract type
if "Contract" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "contract_vs_churn.png", dpi=150)
    plt.close()

print("\nEDA graphs saved.")


# ==================================================
# 5. SPLIT DATA INTO TRAIN AND TEST
# ==================================================

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# 6. PREPROCESSING PIPELINE
# ==================================================

num_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols),
])


# ==================================================
# 7. TRAIN MACHINE LEARNING MODELS
# ==================================================

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ]),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\n========== TRAINING {name} ==========")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

    trained_models[name] = model

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{filename}.png",
        dpi=150,
    )
    plt.close()


# ==================================================
# 8. MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results).sort_values(
    by="ROC-AUC", ascending=False
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv", index=False
)

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

print("\nBest model by ROC-AUC:", best_model_name)

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_churn_model.joblib",
)


# ==================================================
# 9. SHAP EXPLAINABILITY
# ==================================================

# Explain Random Forest predictions with TreeExplainer.
rf_pipeline = trained_models["Random Forest"]
fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_classifier = rf_pipeline.named_steps["classifier"]

X_test_transformed = fitted_preprocessor.transform(X_test)

# Use a sample to keep SHAP calculations manageable.
sample_size = min(200, X_test_transformed.shape[0])

rng = np.random.default_rng(RANDOM_STATE)
sample_indices = rng.choice(
    X_test_transformed.shape[0],
    size=sample_size,
    replace=False,
)

X_shap = X_test_transformed[sample_indices]

if hasattr(X_shap, "toarray"):
    X_shap = X_shap.toarray()
else:
    X_shap = np.asarray(X_shap)

feature_names = fitted_preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(rf_classifier)
raw_shap_values = explainer.shap_values(X_shap)

# Support common SHAP output formats.
if isinstance(raw_shap_values, list):
    churn_shap_values = raw_shap_values[1]
elif np.asarray(raw_shap_values).ndim == 3:
    churn_shap_values = np.asarray(raw_shap_values)[:, :, 1]
else:
    churn_shap_values = np.asarray(raw_shap_values)

# Global SHAP explanation
shap.summary_plot(
    churn_shap_values,
    X_shap,
    feature_names=feature_names,
    max_display=15,
    show=False,
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# Feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.abs(churn_shap_values).mean(axis=0),
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False,
)

print("\nTop SHAP features:")
print(importance_df.head(10).to_string(index=False))

# Explain one individual customer
base_values = np.asarray(explainer.expected_value)

if base_values.ndim == 0:
    base_value = float(base_values)
else:
    base_value = float(base_values.reshape(-1)[-1])

customer_explanation = shap.Explanation(
    values=churn_shap_values[0],
    base_values=base_value,
    data=X_shap[0],
    feature_names=list(feature_names),
)

shap.plots.waterfall(customer_explanation, show=False)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_customer_explanation.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()


# ==================================================
# 10. SAMPLE CUSTOMER PREDICTION
# ==================================================

sample_customer = X_test.iloc[[0]]

rf_probability = rf_pipeline.predict_proba(
    sample_customer
)[0, 1]

rf_prediction = rf_pipeline.predict(sample_customer)[0]

print("\n========== SAMPLE CUSTOMER PREDICTION ==========")
print(f"Churn probability: {rf_probability:.2%}")
print(
    "Predicted class:",
    "Churn" if rf_prediction == 1 else "No Churn",
)

print("\nAll output files saved in:", OUTPUT_DIR)
print("PROJECT COMPLETED SUCCESSFULLY!")
