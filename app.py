import streamlit as st
import requests
import subprocess
import os

API_BASE_URL = "https://inventory-forecast-api-1070349950078.asia-south1.run.app"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

PRODUCT_IDS = [
    20, 120, 150, 153, 165,
    298, 310, 375, 437, 448
]

@st.cache_resource
def get_identity_token():
    return subprocess.check_output(
        ["gcloud", "auth", "print-identity-token"],
        shell=True,
        text=True
    ).strip()

st.set_page_config(
    page_title="Inventory Forecast Dashboard",
    layout="centered"
)

st.title(" Inventory Forecast Dashboard")
st.caption("Demand prediction using ML models deployed on Cloud Run")

tab1, tab2 = st.tabs([" Prediction", " Visualization"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        actual_product_id = st.selectbox(
            "Select Product ID",
            options=PRODUCT_IDS
        )

    with col2:
        selected_date = st.date_input(
            "Select Forecast Date",
            help="Format: YYYY-MM-DD"
        )

    if st.button(" Run Forecast"):
        with st.spinner("Predicting demand..."):
            try:
                token = get_identity_token()

                headers = {
                    "Authorization": f"Bearer {token}"
                }

                params = {
                    "product_id": actual_product_id,
                    "date": str(selected_date)
                }

                response = requests.get(
                    PREDICT_ENDPOINT,
                    headers=headers,
                    params=params,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction completed successfully")

                    st.metric(
                        label=" Predicted Quantity",
                        value=result["predicted_quantity"]
                    )

                    st.markdown(f"""
                    **Product ID:** {result['product_id']}  
                    **Date:** {result['date']}  
                    **Model Used:** `{result['model_used']}`
                    """)

                    st.session_state["last_product_id"] = actual_product_id

                else:
                    st.error(f"API Error {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error("Unexpected error occurred")
                st.code(str(e))

with tab2:
    st.subheader(" Sales Trend & Forecast")

    if "last_product_id" in st.session_state:
        product_id = st.session_state["last_product_id"]
        graph_path = f"graphs/product_{product_id}.png"

        if os.path.exists(graph_path):
            with st.expander("View Forecast Graph", expanded=True):
                st.image(graph_path, width=800)
        else:
            st.warning("Graph not available for the selected product.")
    else:
        st.info("Run a prediction first to view the graph.")
