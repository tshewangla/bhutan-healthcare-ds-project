import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

st.set_page_config(page_title="Bhutan Life Expectancy Analytics", layout="wide")

# Path to default cleaned dataset (bundled with the repo)
DEFAULT_DATA_PATH = "data/processed/life_expectancy_btn_clean.csv"

# -----------------------
# Helpers for session state
# -----------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.feature_cols = None
    st.session_state.target_col = None
    st.session_state.problem_type = None  # "classification" or "regression"


# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Upload & Clean Data", "EDA", "Train Model", "Predict"]
)


# -----------------------
# Page: Overview
# -----------------------
if page == "Overview":
    st.title("Bhutan Life Expectancy Analysis & Prediction")

    st.markdown(
        """
        This Streamlit app is part of a data science project analysing
        **life expectancy in Bhutan (2000–2021)** using WHO data.

        ### What you can do in this app:
        1. Load a built-in cleaned dataset (Bhutan life expectancy).
        2. Or upload your own healthcare CSV file.
        3. Perform basic automatic cleaning (duplicates + missing values).
        4. Explore the dataset (summary statistics, correlations).
        5. Train a machine learning model (regression or classification).
        6. Make predictions from user inputs.

        For the assignment, the main focus dataset is
        **Life expectancy at birth (years)** for Bhutan, but the app can
        also work with other tabular health datasets.
        """
    )

# -----------------------
# Page: Upload & Clean Data
# -----------------------
elif page == "Upload & Clean Data":
    st.title("1. Upload & Clean Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file (e.g. WHO indicators, health metrics, etc.)",
        type=["csv"]
    )

    st.write("Alternatively, use the default Bhutan life expectancy dataset:")

    if st.button("Load default Bhutan Life Expectancy Data"):
        if os.path.exists(DEFAULT_DATA_PATH):
            df_default = pd.read_csv(DEFAULT_DATA_PATH)
            st.session_state.df = df_default
            st.success("Loaded default Bhutan life expectancy dataset from project files.")
            st.write("### Default Dataset Preview")
            st.write(df_default.head())
            st.info(
                "You can now go to **EDA**, **Train Model**, or **Predict** "
                "using the default dataset."
            )
        else:
            st.error(
                f"Default dataset not found at `{DEFAULT_DATA_PATH}`. "
                "Please make sure the file exists in your repository."
            )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(df.head())

        st.write("Shape:", df.shape)

        # --- Basic cleaning ---
        st.subheader("Basic Cleaning (automatic)")
        df_clean = df.copy()

        # Drop completely duplicated rows
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after = len(df_clean)
        st.write(f"- Removed **{before - after}** duplicate rows")

        # Handle missing values: numeric -> median, others -> mode
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df_clean.columns if c not in num_cols]

        for col in num_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        for col in cat_cols:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(mode_val)

        st.write("### Cleaned Data Preview")
        st.write(df_clean.head())
        st.write("Any remaining missing values per column:")
        st.write(df_clean.isna().sum())

        # Save to session + disk (for assignment structure)
        st.session_state.df = df_clean

        os.makedirs("data/processed", exist_ok=True)
        cleaned_path = "data/processed/cleaned_data.csv"
        df_clean.to_csv(cleaned_path, index=False)
        st.success(f"Cleaned data saved to `{cleaned_path}` inside the project.")

        st.info(
            "Next: go to the **EDA** page to explore the cleaned data, "
            "then **Train Model**."
        )
    else:
        if st.session_state.df is None:
            st.info(
                "Upload a CSV file above or click the button to load the "
                "default Bhutan life expectancy dataset."
            )


# -----------------------
# Page: EDA
# -----------------------
elif page == "EDA":
    st.title("2. Exploratory Data Analysis (EDA)")

    df = st.session_state.df
    if df is None:
        st.warning("No data loaded yet. Go to **Upload & Clean Data** first.")
    else:
        st.subheader("Dataset Overview")
        st.write("Shape:", df.shape)
        st.write(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values per Column")
        st.write(df.isna().sum())

        st.subheader("Column Types")
        st.write(df.dtypes)

        # Simple correlation matrix (numeric only)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            st.subheader("Correlation Matrix (numerical features)")
            corr = df[num_cols].corr()
            st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
        else:
            st.info("Not enough numeric columns to compute a correlation matrix.")


# -----------------------
# Page: Train Model
# -----------------------
elif page == "Train Model":
    st.title("3. Train Machine Learning Model")

    df = st.session_state.df
    if df is None:
        st.warning("No data loaded yet. Go to **Upload & Clean Data** first.")
    else:
        st.write("First, choose which column you want to **predict** (the target).")
        target_col = st.selectbox("Target column", options=df.columns)

        feature_cols = [c for c in df.columns if c != target_col]

        if not feature_cols:
            st.error("You need at least one feature column besides the target.")
        else:
            st.write("Feature columns being used:", feature_cols)

            # Decide problem type
            y = df[target_col]
            if y.dtype == "object" or y.nunique() <= 10:
                problem_type = "classification"
            else:
                problem_type = "regression"

            st.write(f"Detected problem type: **{problem_type}**")

            # One-hot encode categorical variables
            X = df[feature_cols].copy()
            X = pd.get_dummies(X, drop_first=True)

            # Align target
            if problem_type == "classification":
                y = y.astype("category")
            else:
                y = pd.to_numeric(y, errors="coerce")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if st.button("Train Model"):
                if problem_type == "classification":
                    model = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    st.write(f"**Accuracy:** {acc:.3f}")
                    st.write(f"**F1-score (weighted):** {f1:.3f}")
                else:
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"**RMSE:** {rmse:.3f}")
                    st.write(f"**R²:** {r2:.3f}")

                # Save model + metadata
                st.session_state.model = model
                st.session_state.feature_cols = feature_cols
                st.session_state.target_col = target_col
                st.session_state.problem_type = problem_type

                os.makedirs("models", exist_ok=True)
                joblib.dump(
                    {
                        "model": model,
                        "feature_cols": feature_cols,
                        "problem_type": problem_type,
                        "target_col": target_col,
                    },
                    "models/model.pkl",
                )

                st.success(
                    "Model trained and saved to `models/model.pkl`. "
                    "You can now go to the **Predict** page."
                )


# -----------------------
# Page: Predict
# -----------------------
elif page == "Predict":
    st.title("4. Make Predictions")

    df = st.session_state.df
    model = st.session_state.model
    feature_cols = st.session_state.feature_cols
    problem_type = st.session_state.problem_type

    if df is None or model is None or feature_cols is None:
        st.warning(
            "No model or data found. "
            "Please upload/load data and train a model first."
        )
    else:
        st.write("Provide input values for the features below:")

        # Build input widgets from original dataframe (not one-hot encoded)
        input_data = {}
        for col in feature_cols:
            if df[col].dtype == "object":
                options = df[col].dropna().unique().tolist()
                if not options:
                    input_val = ""
                else:
                    input_val = st.selectbox(col, options=options)
            else:
                # Use reasonable defaults based on data
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_val = st.number_input(
                    col, value=mean_val, min_value=min_val, max_value=max_val
                )
            input_data[col] = input_val

        if st.button("Predict"):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply same one-hot encoding as training
            all_data = pd.concat(
                [df[feature_cols], input_df],
                axis=0
            )
            all_encoded = pd.get_dummies(all_data, drop_first=True)

            input_encoded = all_encoded.tail(1)

            # Align columns with training model
            train_X = pd.get_dummies(df[feature_cols], drop_first=True)
            missing_cols = [c for c in train_X.columns if c not in input_encoded.columns]
            for c in missing_cols:
                input_encoded[c] = 0
            input_encoded = input_encoded[train_X.columns]

            pred = model.predict(input_encoded)[0]

            if problem_type == "classification":
                st.success(f"Predicted class for the target: **{pred}**")
            else:
                st.success(f"Predicted value for the target: **{pred:.3f}**")

            st.info(
                "For your report, you can explain what this prediction means in the "
                "context of Bhutanese healthcare (e.g. predicted life expectancy or "
                "other health indicator value)."
            )
