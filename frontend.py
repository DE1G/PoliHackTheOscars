import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from forecast_full import load_model, predict

# Set page config
st.set_page_config(page_title="Forecast Dashboard", layout="wide")

st.title("📈 Forecasted Volume Dashboard")

# Load validation dataset
@st.cache_data
def load_validation_data():
    return pd.read_csv("data/val/forecast_full.csv", parse_dates=["date"])

df_val = load_validation_data()

# Show category selector
unique_categories = df_val["category"].dropna().unique()
selected_category = st.selectbox("Select Category", sorted(unique_categories))

# Filter data
df_filtered = df_val[df_val["category"] == selected_category].copy()

if df_filtered.empty:
    st.warning("No data available for this category.")
    st.stop()

# Load model and make predictions
model = load_model()
try:
    df_processed, predictions = predict(model, df_filtered)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Ensure date is datetime and sort
df_processed["date"] = pd.to_datetime(df_processed["date"])
df_processed["predicted_volume"] = predictions
df_processed = df_processed.sort_values("date")

# Optional: average per day (to make it smoother if there's a lot of variation)
plot_df = df_processed.groupby("date").agg({
    "actual_volume": "mean",
    "predicted_volume": "mean"
}).reset_index()

# Plot
st.subheader(f"📊 Smoothed Actual vs Predicted Volume for '{selected_category}'")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["date"], plot_df["actual_volume"], label="Actual", color="blue", alpha=0.7)
ax.plot(plot_df["date"], plot_df["predicted_volume"], label="Predicted", color="orange", alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
ax.set_title(f"Smoothed Forecast for Category: {selected_category}")
ax.legend()
ax.grid(True)

# Improve x-axis ticks
fig.autofmt_xdate()
st.pyplot(fig)

# Optional: download data
st.download_button(
    label="📥 Download Prediction CSV",
    data=plot_df.to_csv(index=False),
    file_name=f"forecast_{selected_category}.csv",
    mime="text/csv"
)
