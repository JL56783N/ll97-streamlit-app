import streamlit as st
import pandas as pd
import joblib
# Load models
fined_model = joblib.load("fined_model.joblib")
# Page config
st.title("🔍 NYC LL97 Compliance Prediction Dashboard")
# Tabs
tabs = st.tabs(["🏢 Predictions", "📊 Insights", "🧠 How It Works", "📋 Model Details"])
# -----------------------------
# Tab 1: Predictions
# -----------------------------
with tabs[0]:
    st.header("Predict: Will This Building Be Fined or Pay the Fine?")
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        energy_star_score = col2.number_input("Energy Star Score", value=75)
        year = col1.selectbox("Calendar Year", [2020, 2021, 2022, 2023])
        site_eui = col3.number_input("Site EUI (kBtu/ft²)", value=150.0)
        ghg = st.number_input("Total GHG Emissions (Metric Tons CO2e)", value=500.0)
        property_type = st.selectbox("Property Type", [
        "Office", "Multifamily Housing", "K-12 School", "Hotel",
        "Warehouse", "Retail Store", "Hospital (General Medical & Surgical)",
        "Financial Office", "Distribution Center"
        ])
        submitted = st.form_submit_button("Run Prediction")
        if submitted:
            # Define input_df with enforced types and order
            input_df = input_df[["property_type", "calendar_year", "energy_star_score", "site_eui", "ghg_emissions"]]
            # Run predictions
