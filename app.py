import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------- Helpers (alag alag Streamlit versions par chalne ke liye) ----------
def show_df(data):
    try:
        st.dataframe(data, width="stretch")
    except Exception:
        st.dataframe(data, use_container_width=True)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def explain_prediction(model, customer_df, customer, categorical_cols, top_n=5):
    """
    Logistic Regression ke coefficients se har feature ka contribution nikalta hai.
    contribution = (preprocess ki hui value) x (model coefficient)
    Positive = churn ka risk barhata hai, Negative = risk kam karta hai.
    Return: DataFrame (Factor, Impact, Effect)
    """
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    transformed = pre.transform(customer_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    contributions = np.asarray(transformed)[0] * clf.coef_[0]
    feature_names = pre.get_feature_names_out()

    # One-hot columns ko wapas original column mein jama karein
    sorted_cats = sorted(categorical_cols, key=len, reverse=True)
    grouped = {}
    for name, value in zip(feature_names, contributions):
        if name.startswith("num__"):
            original = name[len("num__"):]
        else:
            stripped = name[len("cat__"):]
            original = next(
                (c for c in sorted_cats if stripped.startswith(c + "_")),
                stripped
            )
        grouped[original] = grouped.get(original, 0.0) + float(value)

    rows = []
    for column, impact in grouped.items():
        if abs(impact) < 0.01:
            continue
        rows.append({
            "Factor": f"{column} = {customer.get(column, '')}",
            "Impact": impact,
            "Effect": (
                "Increases churn risk" if impact > 0 else "Decreases churn risk"
            )
        })

    result = pd.DataFrame(rows, columns=["Factor", "Impact", "Effect"])
    if result.empty:
        return result

    up = result[result["Impact"] > 0].nlargest(top_n, "Impact")
    down = result[result["Impact"] < 0].nsmallest(top_n, "Impact")
    return pd.concat([up, down]).reset_index(drop=True)


st.title("📊 Customer Churn Prediction System")
st.caption("Telecom Customer Analytics & Machine Learning Dashboard")

file_path = Path(__file__).parent / "WA_Fn-UseC_-Telco-customer-Churn.csv"

if not file_path.exists():
    st.error("CSV file app.py ke folder mein nahi mili.")
    st.stop()

df = pd.read_csv(file_path)

# ---------- Data cleaning ----------
df.columns = df.columns.str.strip()

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

if "Churn" not in df.columns:
    st.error("Dataset mein Churn column nahi hai.")
    st.stop()

df = df.dropna(subset=["Churn"])
df["Churn"] = df["Churn"].astype(str).str.strip()

st.success("Dataset successfully loaded!")

# ---------- Sidebar navigation ----------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset", "Train Model", "Predict Churn"]
)

# ======================= DASHBOARD =======================
if page == "Dashboard":
    st.header("Dashboard Overview")

    total_customers = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = int((df["Churn"] == "No").sum())
    churn_rate = churned / total_customers * 100 if total_customers else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    churn_counts = (
        df["Churn"].value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Churn Distribution")
        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.45,
            title="Churn vs Retained"
        )
        show_chart(fig)

    with col2:
        st.subheader("Churn Count")
        fig = px.bar(
            churn_counts,
            x="Churn",
            y="Customers",
            color="Churn",
            title="Number of Customers"
        )
        show_chart(fig)

    col3, col4 = st.columns(2)

    with col3:
        if "Contract" in df.columns:
            st.subheader("Customers by Contract")
            contract_df = (
                df.groupby(["Contract", "Churn"])
                .size()
                .reset_index(name="Customers")
            )
            fig = px.bar(
                contract_df,
                x="Contract",
                y="Customers",
                color="Churn",
                barmode="group",
                title="Contract Type and Churn"
            )
            show_chart(fig)

    with col4:
        if "MonthlyCharges" in df.columns:
            st.subheader("Monthly Charges")
            fig = px.histogram(
                df,
                x="MonthlyCharges",
                color="Churn",
                nbins=30,
                barmode="overlay",
                title="Monthly Charges by Churn"
            )
            show_chart(fig)

# ======================= DATASET =======================
elif page == "Dataset":
    st.header("Dataset Explorer")

    c1, c2 = st.columns(2)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))

    st.subheader("Dataset Preview")
    show_df(df.head(20))

    if st.checkbox("Show Dataset Statistics"):
        show_df(df.describe(include="all").T.astype(str))

    if st.checkbox("Show Missing Values"):
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        show_df(missing)

# ======================= TRAIN MODEL =======================
elif page == "Train Model":
    st.header("🤖 Train Machine Learning Model")
    st.write("Logistic Regression model ko dataset par train karein.")

    if st.button("🚀 Train Model", type="primary"):
        X = df.drop(columns=["Churn"]).copy()
        y = df["Churn"].map({"No": 0, "Yes": 1})

        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid].astype(int)

        if y.nunique() < 2:
            st.error(
                "Training ke liye Churn mein Yes aur No dono classes honi chahiye."
            )
            st.stop()

        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        with st.spinner("Model training ho raha hai..."):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

        st.session_state["churn_model"] = model
        st.session_state["model_columns"] = X.columns.tolist()
        st.session_state["model_accuracy"] = accuracy
        st.session_state["categorical_cols"] = categorical_cols

        st.success("Model successfully trained!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Training Records", len(X_train))
        c2.metric("Testing Records", len(X_test))
        c3.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No Churn", "Actual Churn"],
            columns=["Predicted No Churn", "Predicted Churn"]
        )
        show_df(cm_df)

        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0
        )
        show_df(pd.DataFrame(report).T.round(3))

    if "churn_model" in st.session_state:
        st.info(
            f"Trained model available. Test accuracy: "
            f"{st.session_state['model_accuracy'] * 100:.2f}%"
        )

# ======================= PREDICT CHURN =======================
elif page == "Predict Churn":
    st.header("🔍 Predict Customer Churn")

    if "churn_model" not in st.session_state:
        st.warning("Pehle Train Model page par ja kar model train karein.")
        st.stop()

    model = st.session_state["churn_model"]
    columns = st.session_state["model_columns"]

    with st.form("prediction_form"):
        st.subheader("Enter Customer Details")
        customer = {}

        left, right = st.columns(2)

        for i, column in enumerate(columns):
            series = df[column] if column in df.columns else None
            box = left if i % 2 == 0 else right

            with box:
                if column == "TotalCharges":
                    customer[column] = st.number_input(
                        "Total Charges", min_value=0.0, value=100.0,
                        key=f"in_{column}"
                    )

                elif column == "MonthlyCharges":
                    customer[column] = st.number_input(
                        "Monthly Charges", min_value=0.0, value=70.0,
                        key=f"in_{column}"
                    )

                elif column == "tenure":
                    customer[column] = st.number_input(
                        "Tenure (months)", min_value=0, max_value=100,
                        value=12, key=f"in_{column}"
                    )

                # Koi bhi numeric column jo sirf 0/1 ho -> Yes/No dropdown
                elif (
                    series is not None
                    and pd.api.types.is_numeric_dtype(series)
                    and set(series.dropna().unique()) <= {0, 1}
                ):
                    label = (
                        "Senior Citizen" if column == "SeniorCitizen" else column
                    )
                    choice = st.selectbox(
                        label, ["No", "Yes"], key=f"in_{column}"
                    )
                    customer[column] = 1 if choice == "Yes" else 0

                # Text columns -> dropdown
                elif series is not None and not pd.api.types.is_numeric_dtype(series):
                    options = series.dropna().astype(str).unique().tolist()
                    if options:
                        customer[column] = st.selectbox(
                            column, options, key=f"in_{column}"
                        )
                    else:
                        customer[column] = ""

                # Baaqi numeric columns
                elif series is not None:
                    median_value = pd.to_numeric(series, errors="coerce").median()
                    default_value = (
                        float(median_value) if pd.notna(median_value) else 0.0
                    )
                    customer[column] = st.number_input(
                        column, value=default_value, key=f"in_{column}"
                    )

                else:
                    customer[column] = 0

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        customer_df = pd.DataFrame([customer], columns=columns)

        prediction = int(model.predict(customer_df)[0])
        probability = float(model.predict_proba(customer_df)[0][1])

        if prediction == 1:
            st.error("Prediction: Customer may Churn")
        else:
            st.success("Prediction: Customer likely to Stay")

        st.metric("Estimated Churn Probability", f"{probability * 100:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "Yeh model estimate hai, customer ke future behavior "
            "ki guarantee nahi."
        )

        # ---------- NEW: Top contributing factors ----------
        st.divider()
        st.subheader("🔎 Top Contributing Factors")

        try:
            factors = explain_prediction(
                model,
                customer_df,
                customer,
                st.session_state.get("categorical_cols", [])
            )
        except Exception as e:
            factors = None
            st.warning(f"Factors nikalne mein masla aaya: {e}")

        if factors is not None and not factors.empty:
            plot_df = factors.sort_values("Impact")
            fig = px.bar(
                plot_df,
                x="Impact",
                y="Factor",
                color="Effect",
                orientation="h",
                color_discrete_map={
                    "Increases churn risk": "#e74c3c",
                    "Decreases churn risk": "#2ecc71"
                },
                title="Kaunsi cheezein is customer ka churn risk badha ya ghata rahi hain"
            )
            show_chart(fig)

            risk_up = factors[factors["Impact"] > 0]
            risk_down = factors[factors["Impact"] < 0]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🔴 Risk barhane wali cheezein**")
                if risk_up.empty:
                    st.write("Koi nahi")
                for _, row in risk_up.iterrows():
                    st.write(f"- {row['Factor']}")
            with col_b:
                st.markdown("**🟢 Risk kam karne wali cheezein**")
                if risk_down.empty:
                    st.write("Koi nahi")
                for _, row in risk_down.iterrows():
                    st.write(f"- {row['Factor']}")

            st.caption(
                "Yeh factors model ke patterns dikhate hain (Logistic Regression "
                "coefficients se), churn ki pakki wajah ki guarantee nahi."
            )
        elif factors is not None:
            st.info("Is customer ke liye koi numaya factor nahi mila.")