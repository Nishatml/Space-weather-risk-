import streamlit as st
import pandas as pd
from datetime import datetime
from config import init_groq_client
from utils import fetch_dscovr_data, run_ml_inference
from generator import generate_incident_report

st.set_page_config(page_title="Space Weather Early Warning System", page_icon="", layout="wide")

client = init_groq_client()

st.title(" Space Weather Early Warning System")
st.markdown("Real-time telemetry ingestion from NOAA DSCOVR satellite with ML-driven Kp forecasting and LLM incident reporting.")

if not client:
    st.warning("GROQ_API_KEY missing! Ensure it is defined in Streamlit secrets.")

# Fetch telemetry data
telemetry_data = fetch_dscovr_data()

if telemetry_data is not None and not telemetry_data.empty:
    metrics = run_ml_inference(telemetry_data)

    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Kp Index", metrics["kp_index"])
    col2.metric("Geomagnetic Storm Prob", f"{metrics['storm_prob']}%")
    col3.metric("IMF Bz (nT)", metrics["bz"])
    col4.metric("Solar Wind Speed (km/s)", metrics["speed"])

    st.markdown("---")
    st.subheader(" Live Telemetry Data")
    st.dataframe(telemetry_data, use_container_width=True)

    # AI Incident Report Generator
    st.markdown("---")
    st.subheader("AI Incident Report Generator")
    if st.button("Generate AI Incident Report"):
        if client:
            with st.spinner("Generating AI analysis..."):
                report = generate_incident_report(client, metrics)
                st.markdown(report)
        else:
            st.error("Cannot generate report: GROQ_API_KEY is not configured.")
else:
    st.error("Failed to load telemetry data.")
    metrics = run_ml_inference(telemetry_data)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Kp Index", metrics['kp_index'])
    col2.metric("Storm Probability", f"{metrics['storm_prob']}%")
    col3.metric("IMF Bz", f"{metrics['bz']} nT")
    col4.metric("Solar Wind Speed", f"{metrics['speed']} km/s")
    
    st.divider()
    
    if metrics['kp_index'] >= 5.0:
        st.error(f" GEOMAGNETIC STORM WARNING: Kp Index forecasted at {metrics['kp_index']}")
    else:
        st.success(" Geomagnetic Conditions are currently Normal / Quiet.")
        
    with st.expander("Show Raw DSCOVR Telemetry Data"):
        st.dataframe(telemetry_data.tail(10))
        
    st.subheader(" Agentic AI Incident & Advisory Report")
    if st.button("Generate Technical Report"):
        with st.spinner("Analyzing telemetry via Groq LLM..."):
            report = generate_incident_report(client, metrics)
            st.markdown(report)
            
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name=f"space_weather_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
            