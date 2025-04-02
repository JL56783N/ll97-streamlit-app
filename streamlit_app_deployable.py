import streamlit as st
import pandas as pd
import joblib
# Load models
fined_model = joblib.load("fined_model.joblib")
paid_model = joblib.load("paid_model.joblib")
# Page config
st.set_page_config(page_title="LL97 Compliance Predictor", layout="wide")
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
year = col1.selectbox("Calendar Year", [2020, 2021, 2022, 2023])
energy_star = col2.slider("ENERGY STAR Score", 1, 100, 75)
site_eui = col3.number_input("Site EUI (kBtu/ft²)", value=150.0)
ghg = st.number_input("Total GHG Emissions (Metric Tons CO2e)", value=500.0)
property_type = st.selectbox("Property Type", [
"Office", "Multifamily Housing", "K-12 School", "Hotel",
"Warehouse", "Retail Store", "Hospital (General Medical & Surgical)",
"Financial Office", "Distribution Center"
])
submitted = st.form_submit_button("Run Prediction")
if submitted:
input_df = pd.DataFrame([{
"calendar_year": year,
"energy_star_score": energy_star,
"site_eui": site_eui,
"ghg_emissions": ghg,
"property_type": property_type
}])
# Run predictions
fined_pred = fined_model.predict(input_df)[0]
fined_prob = fined_model.predict_proba(input_df)[0][1]
if fined_pred:
st.warning(f"⚠️ Prediction: This building will likely be **fined**. (Confidence: {fined_prob:.2%})")
paid_pred = paid_model.predict(input_df)[0]
paid_prob = paid_model.predict_proba(input_df)[0][1]
if paid_pred:
st.success(f"💸 And it's likely the fine will be **paid**. (Confidence: {paid_prob:.2%})")
else:
st.error(f"❌ But it's likely the fine will **not be paid**. (Confidence: {1 - paid_prob:.2%})")
else:
st.success(f"✅ Prediction: This building is unlikely to be fined. (Confidence: {1 - fined_prob:.2%})")
# -----------------------------
# Tab 2: Insights
# -----------------------------
with tabs[1]:
st.header("📊 Summary Insights (Static or Loadable)")
st.markdown("""
This section can display summary statistics, property-type comparisons,
borough breakdowns, or scorecard charts using uploaded or merged data.
(You can build this out with Altair, Plotly, or Matplotlib if needed.)
""")
# -----------------------------
# Tab 3: How It Works
# -----------------------------
with tabs[2]:
st.header("🧠 How This Dashboard Works")
st.markdown("""
**LL97 Compliance Prediction** combines public NYC building emissions data (LL84)
with violation outcomes (DOB ECB) to predict:
1. Whether a building will be fined for non-compliance
2. If fined, whether the building will actually pay
The models are built using Random Forest Classifiers trained on energy usage,
GHG emissions, ENERGY STAR scores, and building types.
Features include:
- Interactive input form
- Real-time predictions
- Model confidence scores
- Business-readable results
""")
# -----------------------------
# Tab 4: Model Details
# -----------------------------
with tabs[3]:
st.header("📋 Model Pipeline & Assumptions")
st.markdown("""
**Fined Model:**
- Trained on ~5,000 buildings
- Input features: ENERGY STAR score, GHG, EUI, property type, year
- Output: Probability of being fined under LL97
**Paid Model:**
- Subset: Buildings that were fined
- Output: Probability the fine will be paid
**Preprocessing:**
- OneHotEncoding on property type
- No normalization needed (tree-based models)
Models exported as `.joblib` and loaded into Streamlit at runtime.
""")
{
cell_type": "code
id": "047b9a55-4ae1-4523-a293-c33026c7c019