# bhutan-healthcare-ds-project
“Bhutan Healthcare Data Science Group Project”

Bhutan Life Expectancy Analysis & Prediction

Data Science Group Project – Completed by Group 1.

This project analyses life expectancy trends in Bhutan from 2000–2021 using official WHO data.
It includes data cleaning, exploratory data analysis, feature engineering, machine learning modeling, and deployment using a fully interactive Streamlit web app.

bhutan-healthcare-ds-project/
│
├── data/
│   ├── raw/                     # Original WHO dataset
│   └── processed/               # Cleaned and feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb   # Cleanup of life expectancy data
│   ├── 02_EDA.ipynb             # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb        # Regression model training
│
├── models/
│   └── model.pkl                # RandomForest regression model
│
├── app/
│   └── streamlit_app.py         # Interactive Streamlit dashboard
│
├── docs/
│   └── Project_details.md       # Assignment brief
│
├── requirements.txt
└── README.md
