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

# Load your historical data (replace with your actual data loading)
# Assuming 'df' contains 'Rice Price', 'temperature', 'inflation', and date index
# and is already cleaned and preprocessed as in your previous code.
# Example:
# data = {'Rice Price': [100, 102, 105, 103, 106, 108],
#         'temperature': [25, 26, 27, 24, 28, 29],
#         'inflation': [2, 2.5, 3, 2.8, 3.2, 3.5]}
# index = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01'])
# df = pd.DataFrame(data, index=index)
df = pd.read_csv("Market Price data - 2007 to 2023.csv")

# 1. Train ARIMA models for temperature and inflation
def train_exog_arima(data, target_variable, order=(1, 2, 2)):
    """Trains an ARIMA model for a given exogenous variable."""
    model = ARIMA(data[target_variable], order=order)
    model_fit = model.fit()
    return model_fit

temp_model = train_exog_arima(df, 'Teamperature')
infl_model = train_exog_arima(df, 'Inflation')

# 2. Streamlit App
st.title("Rice Price Prediction")
st.subheader("Predict the price of rice for a future date.")

# Get the prediction date from the user
prediction_date_str = st.date_input("Select the prediction date", datetime(2025, 11, 1))
prediction_date = pd.to_datetime(prediction_date_str)

if st.button("Predict Price"):
    # Calculate the number of periods to forecast
    last_train_date = df.index[-1]
    n_periods = (prediction_date - last_train_date).days  # Get number of days
    if n_periods <= 0:
        st.error("Prediction date must be after the last date in the training data.")
        st.stop()

    # 3. Generate future dates for exogenous variable predictions
    future_dates = pd.date_range(start=last_train_date + 1, periods=n_periods, freq='M') # Daily Frequency

    # 4. Predict future temperature and inflation
    temp_predictions = temp_model.forecast(steps=n_periods)
    infl_predictions = infl_model.forecast(steps=n_periods)

    # Create a DataFrame for the exogenous variable predictions
    future_exog_df = pd.DataFrame({'temperature': temp_predictions, 'inflation': infl_predictions}, index=future_dates)

    # 5. Prepare exogenous variables for rice price prediction
    # Create the lagged features based on the predicted temperature and inflation
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
