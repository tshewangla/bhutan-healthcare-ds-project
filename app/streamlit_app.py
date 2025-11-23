import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

# ------------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Bhutan Life Expectancy Analysis & Prediction",
    layout="wide",
)

# Path to your default cleaned dataset (already in your repo)
DEFAULT_DATA_PATH = "data/processed/life_expectancy_btn_clean.csv"

# ------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.feature_cols = None
    st.session_state.target_col = None
    st.session_state.problem_type = None  # "classification" or "regression"


# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Upload & Load Data",
        "EDA (Analysis)",
        "Train Model",
        "Predict",
    ],
)


# ------------------------------------------------------------------
# PAGE: OVERVIEW
# ------------------------------------------------------------------
if page == "Overview":
    st.title("Bhutan Life Expectancy Analysis & Prediction")

    st.markdown(
        """
        This Streamlit app is part of a data science project analysing **life expectancy in Bhutan (2000–2021)** 
        using World Health Organization (WHO) data.

        The project follows a complete data science workflow:

        - ✅ Data collection (WHO / HDX Bhutan health indicators)
        - ✅ Data cleaning and filtering to **life expectancy at birth (years)**
        - ✅ Exploratory Data Analysis (EDA) by year and sex
        - ✅ Feature engineering (time index, confidence interval span, moving averages)
        - ✅ Machine learning modeling (Random Forest regression)
        - ✅ Deployment as this interactive web app

        In this app you can:

        1. **Load** the default cleaned life expectancy dataset for Bhutan  
        2. Or **upload your own CSV** (e.g. other Bhutan health indicators)  
        3. Run **EDA** to understand trends and summary statistics  
        4. **Train a machine learning model** (regression or classification is auto-detected)  
        5. Use the trained model to **make predictions** from user input  

        For the assignment report, the main focus is the **life expectancy at birth (years)** indicator for Bhutan.
        """
    )

# ------------------------------------------------------------------
# PAGE: UPLOAD & LOAD DATA
# ------------------------------------------------------------------
elif page == "Upload & Load Data":
    st.title("1. Upload & Load Bhutan Health Data")

    st.write(
        """
        You can either **upload a CSV file** (any health-related dataset), 
        or **load the default Bhutan life expectancy dataset** that was cleaned in the notebooks 
        and stored in `data/processed/life_expectancy_btn_clean.csv`.
        """
    )

    # ---- Option A: upload CSV ----
    uploaded_file = st.file_uploader(
        "Upload a CSV file (e.g. WHO Bhutan indicators)",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)

        # --- Basic cleaning (generic, for any dataset) ---
        st.subheader("Basic Cleaning (automatic)")

        df_clean = df.copy()

        # Drop duplicate rows
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
        st.write("Remaining missing values per column:")
        st.write(df_clean.isna().sum())

        # Save to session + local path (for assignment structure)
        st.session_state.df = df_clean

        os.makedirs("data/processed", exist_ok=True)
        cleaned_path = "data/processed/cleaned_data.csv"
        df_clean.to_csv(cleaned_path, index=False)
        st.success(f"Cleaned data saved to `{cleaned_path}` inside the project.")

        st.info(
            "Next: go to the **EDA (Analysis)** page to explore this dataset, "
            "then **Train Model**."
        )

    st.markdown("---")

    # ---- Option B: load default Bhutan life expectancy dataset ----
    st.subheader("Or use the default Bhutan Life Expectancy dataset")

    st.write(
        """
        This option loads the **cleaned life expectancy at birth (years)** dataset for Bhutan, 
        which was prepared in the Jupyter notebooks and saved as:

        `data/processed/life_expectancy_btn_clean.csv`
        """
    )

    load_default = st.button("Load default Bhutan Life Expectancy data")

    if load_default:
        if os.path.exists(DEFAULT_DATA_PATH):
            df_default = pd.read_csv(DEFAULT_DATA_PATH)
            st.session_state.df = df_default

            st.success(
                "Default Bhutan life expectancy dataset loaded successfully!"
            )
            st.write("### Default Dataset Preview")
            st.write(df_default.head())
            st.write("Shape:", df_default.shape)
            st.write("Columns:", list(df_default.columns))
        else:
            st.error(
                f"Could not find `{DEFAULT_DATA_PATH}`. "
                "Make sure the cleaned dataset is in the data/processed folder."
            )


# ------------------------------------------------------------------
# PAGE: EDA (ANALYSIS)
# ------------------------------------------------------------------
elif page == "EDA (Analysis)":
    st.title("2. Exploratory Data Analysis (EDA)")

    df = st.session_state.df

    if df is None:
        st.warning(
            "No data loaded yet. Please go to **Upload & Load Data** first "
            "and either upload a CSV or load the default dataset."
        )
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

        # Special EDA for life expectancy dataset (if columns match)
        if {"year", "life_expectancy"}.issubset(df.columns):
            st.markdown("### Life Expectancy Trends in Bhutan")

            # Ensure correct sorting
            df_sorted = df.sort_values(["year"])

            # If sex column is present, plot by sex
            if "sex" in df.columns:
                st.write("Life expectancy over time, by sex:")
                pivot = (
                    df_sorted.pivot_table(
                        index="year",
                        columns="sex",
                        values="life_expectancy",
                        aggfunc="mean",
                    )
                    .sort_index()
                )
                st.line_chart(pivot)

                st.write(
                    """
                    **Interpretation (for report):**  
                    - Life expectancy in Bhutan has increased steadily between 2000 and 2021.  
                    - Females consistently have higher life expectancy than males.  
                    - The 'Both sexes' line lies between the Male and Female lines.
                    """
                )
            else:
                st.write("Life expectancy over time:")
                ts = (
                    df_sorted.groupby("year")["life_expectancy"]
                    .mean()
                    .reset_index()
                    .set_index("year")
                )
                st.line_chart(ts)

        else:
            st.info(
                "This dataset does not appear to be the life expectancy dataset, "
                "so only generic EDA is shown."
            )


# ------------------------------------------------------------------
# PAGE: TRAIN MODEL
# ------------------------------------------------------------------
elif page == "Train Model":
    st.title("3. Train Machine Learning Model")

    df = st.session_state.df

    if df is None:
        st.warning(
            "No data loaded yet. Please go to **Upload & Load Data** first."
        )
    else:
        st.write(
            """
            Select the **target column** you want to predict.  
            The app will automatically decide whether to treat it as a **classification** 
            (for categories) or **regression** (for numeric values).
            """
        )

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
                        random_state=42,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if problem_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    st.write(f"**Accuracy:** {acc:.3f}")
                    st.write(f"**F1-score (weighted):** {f1:.3f}")
                else:
                    rmse = mean_squared_error(
                        y_test,
                        y_pred,
                        squared=False,
                    )
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"**RMSE:** {rmse:.3f}")
                    st.write(f"**R²:** {r2:.3f}")

                # Save model + metadata for prediction page
                st.session_state.model = model
                st.session_state.feature_cols = list(X.columns)
                st.session_state.target_col = target_col
                st.session_state.problem_type = problem_type

                os.makedirs("models", exist_ok=True)
                joblib.dump(
                    {
                        "model": model,
                        "feature_cols": list(X.columns),
                        "problem_type": problem_type,
                        "target_col": target_col,
                    },
                    "models/model.pkl",
                )

                st.success(
                    "Model trained and saved to `models/model.pkl`. "
                    "You can now go to the **Predict** page."
                )


# ------------------------------------------------------------------
# PAGE: PREDICT
# ------------------------------------------------------------------
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
        st.write(
            """
            Provide input values for each feature below.  
            The app will construct a single-row dataset and use the trained model to generate a prediction.
            """
        )

        # Build input widgets from original dataframe columns (not dummified)
        # We only consider original columns that contributed to feature_cols
        base_feature_cols = [c for c in df.columns if c != st.session_state.target_col]

        input_data = {}
        for col in base_feature_cols:
            if df[col].dtype == "object":
                options = df[col].dropna().unique().tolist()
                if not options:
                    input_val = ""
                else:
                    input_val = st.selectbox(col, options=options)
            else:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_mean = float(df[col].mean())
                input_val = st.number_input(
                    col,
                    value=col_mean,
                    min_value=col_min,
                    max_value=col_max,
                )
            input_data[col] = input_val

        if st.button("Predict"):
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply same one-hot encoding as training
            all_data = pd.concat(
                [df[base_feature_cols], input_df],
                axis=0,
            )
            all_encoded = pd.get_dummies(all_data, drop_first=True)

            input_encoded = all_encoded.tail(1)

            # Ensure columns match training features
            for col in feature_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_cols]

            pred = model.predict(input_encoded)[0]

            if problem_type == "classification":
                st.success(f"Predicted class for **{st.session_state.target_col}**: **{pred}**")
            else:
                st.success(
                    f"Predicted value for **{st.session_state.target_col}**: **{pred:.3f}**"
                )

            st.info(
                "For your report, you can explain what this prediction means in the "
                "context of Bhutanese healthcare (e.g. predicted life expectancy, "
                "or expected value of a health indicator)."
            )
