import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load the trained rice price ARIMAX model
try:
    with open('rice_price_arimax_model.pkl', 'rb') as file:
        loaded_rice_model = pickle.load(file)
except FileNotFoundError:
    st.error("Trained rice price model file not found.")
    st.stop()

# Load your historical data
df = pd.read_csv("Market Price data - 2007 to 2023.csv")
df.index = pd.to_datetime(df['Date'])  # Assuming there's a 'Date' column
df.drop(columns=['Date'], inplace=True)

# 1. Train ARIMA models for temperature and inflation
def train_exog_arima(data, target_variable, order=(1, 1, 1)):
    model = ARIMA(data[target_variable], order=order)
    return model.fit()

temp_model = train_exog_arima(df, 'Teamperature')
infl_model = train_exog_arima(df, 'Inflation')

# 2. Streamlit App
st.title("Rice Price Prediction")
st.subheader("Predict the price of rice for a future date.")

# Get the prediction date from the user
prediction_date_str = st.date_input("Select the prediction date", datetime(2025, 11, 1))
prediction_date = pd.to_datetime(prediction_date_str)

# Calculate the number of months from last_train_date to prediction_date
last_train_date = df.index[-1]
months_to_prediction = (prediction_date.year - last_train_date.year) * 12 + (prediction_date.month - last_train_date.month)

# Check if the selected date is too close to the last date in the dataset
if months_to_prediction <= 0:
    # Automatically adjust the prediction date to be 3 months after the last training date
    st.warning(f"Selected prediction date is too soon. Adjusting to {last_train_date + pd.DateOffset(months=3):%Y-%m-%d}.")
    prediction_date = last_train_date + pd.DateOffset(months=3)
else:
    # Add buffer months for lag feature computation
    future_months = months_to_prediction + 3

if st.button("Predict Price"):
    # Calculate the number of periods to forecast
    n_periods = (prediction_date - last_train_date).days  # Get number of days
    if n_periods <= 0:
        st.error("Prediction date must be after the last date in the training data.")
        st.stop()

    # Generate future monthly dates
    future_dates = pd.date_range(start=last_train_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

    # 4. Predict future temperature and inflation
    temp_predictions = temp_model.forecast(steps=n_periods)
    infl_predictions = infl_model.forecast(steps=n_periods)

    # Create a DataFrame for the exogenous variable predictions
    future_exog_df = pd.DataFrame({'temperature': temp_predictions, 'inflation': infl_predictions}, index=future_dates)

    # 5. Prepare exogenous variables for rice price prediction
    future_exog_df['inflation_lag_1'] = future_exog_df['inflation'].shift(1)
    future_exog_df['inflation_lag_2'] = future_exog_df['inflation'].shift(2)
    future_exog_df['inflation_lag_3'] = future_exog_df['inflation'].shift(3)
    future_exog_df['temperature_lag_1'] = future_exog_df['temperature'].shift(1)
    future_exog_df['temperature_lag_2'] = future_exog_df['temperature'].shift(2)
    future_exog_df['temperature_lag_3'] = future_exog_df['temperature'].shift(3)
    future_exog_df.dropna(inplace=True)  # Drop NaN from lags

    # Ensure we only predict for the target date
    exog_for_rice_prediction = future_exog_df.loc[[prediction_date]]

    # Ensure the order of columns matches the training data
    exog_for_rice_prediction = exog_for_rice_prediction[['temperature', 'inflation', 'inflation_lag_1', 'inflation_lag_2', 'inflation_lag_3', 'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3']]
    # 6. Predict rice price
    rice_prediction = loaded_rice_model.predict(
        start=len(loaded_rice_model.model.endog),
        end=len(loaded_rice_model.model.endog),
        exog=exog_for_rice_prediction
    )

    if len(rice_prediction) > 0:
        predicted_price = rice_prediction.iloc[0]
        st.success(f"The predicted price of rice for {prediction_date.strftime('%B %d, %Y')} is: {predicted_price:.2f}")
    else:
        st.warning("Could not generate rice price prediction.")
