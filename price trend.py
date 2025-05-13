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

# 1. Train ARIMA models for temperature and inflation
def train_exog_arima(data, target_variable, order=(1, 1, 1)):
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
#prediction_date_str = st.date_input("Select the prediction date")
#prediction_date = pd.to_datetime(prediction_date_str).to_period('M')

# Let's say this is your input
import pandas as pd
import streamlit as st

# Sample date input from Streamlit
prediction_date = st.date_input("Select prediction date")  # Returns datetime.date

if st.button("Predict Price"):
    # Calculate number of months to forecast
    pred_period = prediction_date.to_period('M')
    last_period = df.index[-1].to_period('M')
    n_periods = (pred_period - last_period).n

    if n_periods <= 0:
        st.error("Prediction date must be after the last date in the training data.")
        st.stop()

    # Generate future dates (monthly frequency)
    future_dates = pd.date_range(start=last_train_date + pd.offsets.MonthEnd(1), periods=n_periods, freq='M')

    # Forecast temperature and inflation
    temp_predictions = temp_model.forecast(steps=n_periods)
    infl_predictions = infl_model.forecast(steps=n_periods)

    # Create DataFrame
    future_exog_df = pd.DataFrame({'temperature': temp_predictions, 'inflation': infl_predictions}, index=future_dates)

    # Create lags
    future_exog_df['inflation_lag_1'] = future_exog_df['inflation'].shift(1)
    future_exog_df['inflation_lag_2'] = future_exog_df['inflation'].shift(2)
    future_exog_df['inflation_lag_3'] = future_exog_df['inflation'].shift(3)
    future_exog_df['temperature_lag_1'] = future_exog_df['temperature'].shift(1)
    future_exog_df['temperature_lag_2'] = future_exog_df['temperature'].shift(2)
    future_exog_df['temperature_lag_3'] = future_exog_df['temperature'].shift(3)
    future_exog_df.dropna(inplace=True)

    # Align date
    aligned_prediction_date = prediction_date + pd.offsets.MonthEnd(0)

    try:
        exog_for_rice_prediction = future_exog_df.loc[[aligned_prediction_date]]
    except KeyError:
        st.error("Aligned prediction date not found in future exogenous predictions.")
        st.stop()

    # Predict
    exog_for_rice_prediction = exog_for_rice_prediction[['temperature', 'inflation', 'inflation_lag_1',
                                                          'inflation_lag_2', 'inflation_lag_3',
                                                          'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3']]
    rice_prediction = loaded_rice_model.predict(
        start=len(loaded_rice_model.model.endog),
        end=len(loaded_rice_model.model.endog),
        exog=exog_for_rice_prediction
    )

    if len(rice_prediction) > 0:
        predicted_price = rice_prediction.iloc[0]
        st.success(f"The predicted price of rice for {aligned_prediction_date.strftime('%B %d, %Y')} is: {predicted_price:.2f}")
    else:
        st.warning("Could not generate rice price prediction.")

