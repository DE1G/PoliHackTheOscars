import streamlit as st
import matplotlib.pyplot as plt
import forecast_full

st.set_page_config(layout="wide")
st.title("Volume Forecasting Frontend")

full_model = forecast_full.getmodel()

# Select granularity
granularity = st.selectbox("Select granularity", ["total", "state", "category", "full"])
