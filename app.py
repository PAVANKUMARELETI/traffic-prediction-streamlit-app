# ------------------------------------------------------------------
#  Streamlit Web App for Traffic Volume Prediction
# ------------------------------------------------------------------
# This script creates a web-based dashboard using Streamlit to
# predict traffic volume based on user inputs. It loads the
# pre-trained machine learning model and uses it to make
# predictions.
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- Load Model and Columns ---
try:
    model = joblib.load('models/random_forest_model.pkl')
    model_columns = joblib.load('models/training_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the `traffic_prediction_model.py` script first to train and save the model.")
    st.stop()

# --- Load Dataset for Dropdown Options ---
try:
    df_for_options = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    weather_main_options = ['Clear', 'Clouds', 'Rain', 'Mist', 'Snow', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog']
    weather_desc_options = sorted(df_for_options['weather_description'].unique())
except FileNotFoundError:
    st.error("Dataset 'Metro_Interstate_Traffic_Volume.csv' not found for populating dropdowns.")
    st.stop()


# --- App Title and Description ---
st.set_page_config(page_title="Traffic Volume Predictor", layout="wide")
st.title('🚗 Metro Interstate Traffic Volume Prediction')
st.markdown("This app predicts the hourly traffic volume on the I-94 Interstate highway. Input the details below to get a prediction.")


# --- User Input Section ---
st.sidebar.header('Input Features')

temp = st.sidebar.slider('Temperature (°F)', min_value=-30, max_value=120, value=55)
# Convert to Kelvin for the model
temp_k = (temp - 32) * 5/9 + 273.15

rain_1h = st.sidebar.number_input('Rain in last 1 hour (mm)', min_value=0.0, value=0.0, step=0.1, format="%.1f")
snow_1h = st.sidebar.number_input('Snow in last 1 hour (mm)', min_value=0.0, value=0.0, step=0.1, format="%.1f")
clouds_all = st.sidebar.slider('Cloud Cover (%)', min_value=0, max_value=100, value=40)

col1, col2 = st.sidebar.columns(2)
with col1:
    input_date = st.date_input("Date")
with col2:
    input_time = st.time_input("Time")

weather_main = st.sidebar.selectbox('Main Weather Condition', options=weather_main_options)
weather_description = st.sidebar.selectbox('Weather Description', options=weather_desc_options)

# --- Prediction Logic ---
if st.sidebar.button('Predict Traffic Volume'):

    # Combine date and time
    dt_object = datetime.combine(input_date, input_time)

    # Create a dictionary for the input
    input_dict = {
        'temp': temp_k,
        'rain_1h': rain_1h,
        'snow_1h': snow_1h,
        'clouds_all': clouds_all,
        'month': dt_object.month,
        'day': dt_object.day,
        'hour': dt_object.hour,
        'day_of_week': dt_object.weekday(), # Monday=0, Sunday=6
    }

    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    # Weather Main
    for col in [f'weather_main_{w}' for w in weather_main_options if f'weather_main_{w}' in model_columns]:
        input_df[col] = (col == f'weather_main_{weather_main}')

    # Weather Description
    for col in [f'weather_description_{d}' for d in weather_desc_options if f'weather_description_{d}' in model_columns]:
        input_df[col] = (col == f'weather_description_{weather_description}')

    # Align columns with the model's training columns
    final_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(final_df)[0]

    # Display the result
    st.success(f"**Predicted Traffic Volume:** `{int(prediction)}` vehicles per hour")

    with st.expander("Show Prediction Details"):
        st.write("The model used these values for prediction:")
        st.dataframe(final_df)


# --- Add some context/info at the bottom ---
st.markdown("---")
st.write("This app uses a Random Forest Regressor model trained on historical data from the UCI Machine Learning Repository.")

if st.checkbox('Show sample of the original data'):
    st.write(df_for_options.head())