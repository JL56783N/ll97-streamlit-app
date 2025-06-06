{
 "cells": [
  {
   "cell_type": "code",
   "id": "8d37a2fe-192d-427f-894c-66a1c0d34070",
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-01 23:48:56.395 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load models\n",
    "fined_model = joblib.load(\"fined_model.joblib\")\n",
    "paid_model = joblib.load(\"paid_model.joblib\")\n",
    "\n",
    "# Page config\n",
    "st.set_page_config(page_title=\"LL97 Compliance Predictor\", layout=\"wide\")\n",
    "st.title(\"🔍 NYC LL97 Compliance Prediction Dashboard\")\n",
    "\n",
    "# Tabs\n",
    "tabs = st.tabs([\"🏢 Predictions\", \"📊 Insights\", \"🧠 How It Works\", \"📋 Model Details\"])\n",
    "\n",
    "# -----------------------------\n",
    "# Tab 1: Predictions\n",
    "# -----------------------------\n",
    "with tabs[0]:\n",
    "    st.header(\"Predict: Will This Building Be Fined or Pay the Fine?\")\n",
    "\n",
    "    # Input form\n",
    "    with st.form(\"prediction_form\"):\n",
    "        col1, col2, col3 = st.columns(3)\n",
    "        year = col1.selectbox(\"Calendar Year\", [2020, 2021, 2022, 2023])\n",
    "        energy_star = col2.slider(\"ENERGY STAR Score\", 1, 100, 75)\n",
    "        site_eui = col3.number_input(\"Site EUI (kBtu/ft²)\", value=150.0)\n",
    "\n",
    "        ghg = st.number_input(\"Total GHG Emissions (Metric Tons CO2e)\", value=500.0)\n",
    "        property_type = st.selectbox(\"Property Type\", [\n",
    "            \"Office\", \"Multifamily Housing\", \"K-12 School\", \"Hotel\",\n",
    "            \"Warehouse\", \"Retail Store\", \"Hospital (General Medical & Surgical)\",\n",
    "            \"Financial Office\", \"Distribution Center\"\n",
    "        ])\n",
    "\n",
    "        submitted = st.form_submit_button(\"Run Prediction\")\n",
    "\n",
    "    if submitted:\n",
    "        input_df = pd.DataFrame([{\n",
    "            \"calendar_year\": year,\n",
    "            \"energy_star_score\": energy_star,\n",
    "            \"site_eui\": site_eui,\n",
    "            \"ghg_emissions\": ghg,\n",
    "            \"property_type\": property_type\n",
    "        }])\n",
    "\n",
    "        # Run predictions\n",
    "        fined_pred = fined_model.predict(input_df)[0]\n",
    "        fined_prob = fined_model.predict_proba(input_df)[0][1]\n",
    "\n",
    "        if fined_pred:\n",
    "            st.warning(f\"⚠️ Prediction: This building will likely be **fined**. (Confidence: {fined_prob:.2%})\")\n",
    "            paid_pred = paid_model.predict(input_df)[0]\n",
    "            paid_prob = paid_model.predict_proba(input_df)[0][1]\n",
    "\n",
    "            if paid_pred:\n",
    "                st.success(f\"💸 And it's likely the fine will be **paid**. (Confidence: {paid_prob:.2%})\")\n",
    "            else:\n",
    "                st.error(f\"❌ But it's likely the fine will **not be paid**. (Confidence: {1 - paid_prob:.2%})\")\n",
    "        else:\n",
    "            st.success(f\"✅ Prediction: This building is unlikely to be fined. (Confidence: {1 - fined_prob:.2%})\")\n",
    "\n",
    "# -----------------------------\n",
    "# Tab 2: Insights\n",
    "# -----------------------------\n",
    "with tabs[1]:\n",
    "    st.header(\"📊 Summary Insights (Static or Loadable)\")\n",
    "    st.markdown(\"\"\"\n",
    "    This section can display summary statistics, property-type comparisons, \n",
    "    borough breakdowns, or scorecard charts using uploaded or merged data.\n",
    "\n",
    "    (You can build this out with Altair, Plotly, or Matplotlib if needed.)\n",
    "    \"\"\")\n",
    "\n",
    "# -----------------------------\n",
    "# Tab 3: How It Works\n",
    "# -----------------------------\n",
    "with tabs[2]:\n",
    "    st.header(\"🧠 How This Dashboard Works\")\n",
    "    st.markdown(\"\"\"\n",
    "    **LL97 Compliance Prediction** combines public NYC building emissions data (LL84) \n",
    "    with violation outcomes (DOB ECB) to predict:\n",
    "\n",
    "    1. Whether a building will be fined for non-compliance\n",
    "    2. If fined, whether the building will actually pay\n",
    "\n",
    "    The models are built using Random Forest Classifiers trained on energy usage,\n",
    "    GHG emissions, ENERGY STAR scores, and building types.\n",
    "\n",
    "    Features include:\n",
    "    - Interactive input form\n",
    "    - Real-time predictions\n",
    "    - Model confidence scores\n",
    "    - Business-readable results\n",
    "    \"\"\")\n",
    "\n",
    "# -----------------------------\n",
    "# Tab 4: Model Details\n",
    "# -----------------------------\n",
    "with tabs[3]:\n",
    "    st.header(\"📋 Model Pipeline & Assumptions\")\n",
    "    st.markdown(\"\"\"\n",
    "    **Fined Model:**\n",
    "    - Trained on ~5,000 buildings\n",
    "    - Input features: ENERGY STAR score, GHG, EUI, property type, year\n",
    "    - Output: Probability of being fined under LL97\n",
    "\n",
    "    **Paid Model:**\n",
    "    - Subset: Buildings that were fined\n",
    "    - Output: Probability the fine will be paid\n",
    "\n",
    "    **Preprocessing:**\n",
    "    - OneHotEncoding on property type\n",
    "    - No normalization needed (tree-based models)\n",
    "\n",
    "    Models exported as `.joblib` and loaded into Streamlit at runtime.\n",
    "    \"\"\")\n"
  },
  {
   "cell_type": "code",
   "id": "047b9a55-4ae1-4523-a293-c33026c7c019",
   "outputs": [],
   "source": []
 ],
  "kernelspec": {
   "display_name": "anaconda-ai-2024.04-py310",
   "language": "python",
   "name": "conda-env-anaconda-ai-2024.04-py310-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
 },
