
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Analysis & Prediction")
st.caption("Machine Learning | Data Analytics | Educational Project")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

@st.cache_resource
def train_model(target_column):
    df = load_data().copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove rows where the target is missing or invalid.
    if y.nunique() < 2:
        raise ValueError("The target column must have at least two classes.")

    # Exclude columns that cannot provide useful model input.
    X = X.select_dtypes(include=["number", "object", "category", "bool"])
    X = X.loc[:, X.nunique(dropna=True) > 1]

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.columns.difference(numeric_columns)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, list(numeric_columns)),
        ("categorical", categorical_pipeline, list(categorical_columns))
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Stratification keeps class proportions similar where possible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X, y, accuracy, X_test, y_test, predictions


try:
    df = load_data()

    if df.empty or len(df.columns) < 2:
        st.error("heart.csv must contain features and a target column.")
        st.stop()

    preferred = [
        c for c in df.columns
        if c.strip().lower() in [
            "target", "heartdisease", "heart_disease",
            "output", "condition"
        ]
    ]

    with st.sidebar:
        st.header("Settings")
        target = st.selectbox(
            "Select target column",
            options=list(df.columns),
            index=(
                list(df.columns).index(preferred[0])
                if preferred else len(df.columns) - 1
            )
        )

    model, X, y, accuracy, X_test, y_test, predictions = (
        train_model(target)
    )

    st.subheader("Project Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", len(df))
    c2.metric("Dataset Columns", len(df.columns))
    c3.metric("Input Features", len(X.columns))
    c4.metric("Test Accuracy", f"{accuracy:.1%}")

    st.info(
        "Model accuracy is measured on a held-out portion of this dataset. "
        "It does not establish clinical reliability."
    )

    tab1, tab2, tab3 = st.tabs([
        "Dataset Explorer",
        "Visual Analytics",
        "Prediction Demo"
    ])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.write("Missing values by column")
            st.dataframe(
                df.isna().sum().rename("Missing values"),
                use_container_width=True
            )

        with right:
            st.write("Summary statistics")
            st.dataframe(
                df.describe(include="all").T,
                use_container_width=True
            )

    with tab2:
        st.subheader("Target Class Distribution")
        counts = df[target].value_counts(dropna=False)

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Target class")
        ax.set_ylabel("Number of records")
        ax.set_title("Distribution of Target Values")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

        numeric = df.select_dtypes(include="number")
        if len(numeric.columns) >= 2:
            st.subheader("Correlation Heatmap")
            corr = numeric.corr()

            fig, ax = plt.subplots(figsize=(9, 6))
            image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.subheader("Try the Prediction Model")
        st.warning(
            "Demonstration only. Do not use this result to diagnose, "
            "treat, or rule out heart disease."
        )

        with st.form("prediction_form"):
            values = {}

            for col in X.columns:
                series = X[col]

                if (
                    pd.api.types.is_numeric_dtype(series)
                    and series.nunique(dropna=True) > 10
                ):
                    median = float(series.median())
                    minimum = float(series.min())
                    maximum = float(series.max())

                    values[col] = st.number_input(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(median, minimum), maximum)
                    )
                else:
                    options = series.dropna().unique().tolist()

                    if not options:
                        st.error(f"No usable values found for {col}.")
                        st.stop()

                    values[col] = st.selectbox(
                        col,
                        options=options,
                        index=0
                    )

            submitted = st.form_submit_button("Run Demo Prediction")

        if submitted:
            input_df = pd.DataFrame([values], columns=X.columns)
            predicted_class = model.predict(input_df)[0]

            st.subheader("Model Output")
            st.write("Predicted class:", str(predicted_class))

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.named_steps["classifier"].classes_

                probability_df = pd.DataFrame({
                    "Class": [str(c) for c in classes],
                    "Model probability": probabilities
                })

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "These are model outputs, not a personal medical risk "
                "assessment. Seek qualified medical advice for health concerns."
            )

except FileNotFoundError:
    st.error("heart.csv was not found. Put it beside app.py.")

except Exception as error:
    st.error(f"The app could not start: {error}")
    st.info(
        "Check your target column, class counts, data types and missing values."
    )
