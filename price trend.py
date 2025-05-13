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
    if temp_model is None or infl_model is None:
        st.error("Failed to train ARIMA models for temperature and inflation.")
        st.stop()

    try:
        # Convert to Timestamp
        prediction_date = pd.Timestamp(prediction_date)
        last_train_date = pd.to_datetime(df.index[-1])

        # Convert to Period (monthly)
        pred_period = prediction_date.to_period('M')
        last_period = last_train_date.to_period('M')

        # Subtract: this gives an int
        n_periods = (pred_period - last_period).n  # .n ensures it is plain int

    except Exception as e:
        st.error(f"Date processing failed: {e}")
        st.stop()

    # Check if valid
    if n_periods <= 0:
        st.error("Prediction date must be after the last date in the training data.")
        st.stop()

    st.success(f"Forecasting for {n_periods} months...")


    # Now you're safe to proceed with the forecast
    st.success(f"Number of months to forecast: {n_periods}")



    # 3. Generate future dates for exogenous variable predictions
    future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=n_periods, freq='M') # Monthly Frequency

    # 4. Predict future temperature and inflation
    temp_predictions = temp_model.forecast(steps=n_periods)
    infl_predictions = infl_model.forecast(steps=n_periods)

    # Create a DataFrame for the exogenous variable predictions
    future_exog_df = pd.DataFrame({'Teamperature': temp_predictions, 'Inflation': infl_predictions}, index=future_dates)

    # 5. Prepare exogenous variables for rice price prediction
    # Create the lagged features based on the predicted temperature and inflation
    future_exog_df['inflation_lag_1'] = future_exog_df['Inflation'].shift(1)
    future_exog_df['inflation_lag_2'] = future_exog_df['Inflation'].shift(2)
    future_exog_df['inflation_lag_3'] = future_exog_df['Inflation'].shift(3)
    future_exog_df['temperature_lag_1'] = future_exog_df['Teamperature'].shift(1)
    future_exog_df['temperature_lag_2'] = future_exog_df['Teamperature'].shift(2)
    future_exog_df['temperature_lag_3'] = future_exog_df['Teamperature'].shift(3)
    future_exog_df.dropna(inplace=True)  # Drop NaN from lags

    # Ensure we only predict for the target date
    exog_for_rice_prediction = future_exog_df.loc[[prediction_date]]

    # Ensure the order of columns matches the training data
   
    if prediction_date in future_exog_df.index:
        exog_for_rice_prediction = future_exog_df.loc[[prediction_date]]

        # Ensure the order of columns matches the training data
        exog_cols = ['Teamperature', 'Inflation', 'inflation_lag_1', 'inflation_lag_2', 'inflation_lag_3', 'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3']
        if all(col in exog_for_rice_prediction.columns for col in loaded_rice_model.exog_names):
            exog_for_rice_prediction = exog_for_rice_prediction[loaded_rice_model.exog_names]
        else:
            st.error("Error: Exogenous variable columns do not match the trained rice price model.")
            st.stop()

        # 6. Predict rice price
        rice_prediction = loaded_rice_model.predict(
            start=len(loaded_rice_model.model.endog),
            end=len(loaded_rice_model.model.endog),
            exog=exog_for_rice_prediction
        )

        if len(rice_prediction) > 0:
            predicted_price = rice_prediction.iloc[0]
            st.success(f"The predicted price of rice for {prediction_date.strftime('%B, %Y')} is: {predicted_price:.2f}")
        else:
            st.warning("Could not generate rice price prediction.")
    else:
        st.warning(f"Could not find exogenous variable predictions for {prediction_date}.")

