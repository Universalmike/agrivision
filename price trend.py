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

if st.button("Predict Price"):
    # 3. Get the number of months between last date and prediction date
    # 3. Compute number of months to prediction, plus lag buffer
    last_train_date = df.index[-1]
    months_to_prediction = (prediction_date.year - last_train_date.year) * 12 + (prediction_date.month - last_train_date.month)
    
    if months_to_prediction <= 0:
        st.error("Prediction date must be after the last date in the training data.")
        st.stop()
    
    # Add 3 months buffer for lag features
    future_months = months_to_prediction + 3
    
    # 4. Generate future monthly dates
    future_dates = pd.date_range(start=last_train_date + pd.DateOffset(months=1), periods=future_months, freq='MS')

    # 5. Forecast temperature and inflation
    temp_predictions = temp_model.forecast(steps=future_months)
    infl_predictions = infl_model.forecast(steps=future_months)

    # 6. Create future exogenous variable DataFrame
    future_exog_df = pd.DataFrame({
        'Teamperature': temp_predictions,
        'Inflation': infl_predictions
    }, index=future_dates)

    # 7. Create lag features
    for lag in [1, 2, 3]:
        future_exog_df[f'inflation_lag_{lag}'] = future_exog_df['Inflation'].shift(lag)
        future_exog_df[f'temperature_lag_{lag}'] = future_exog_df['Teamperature'].shift(lag)

    # Drop rows with NaNs from lags
    future_exog_df.dropna(inplace=True)

    # Ensure prediction_date exists in index after lag drop
    if prediction_date not in future_exog_df.index:
        st.warning("Not enough data points to compute lag features for the selected date. Try a later date.")
        st.stop()

    # 8. Get exogenous features for prediction date
    exog_for_rice_prediction = future_exog_df.loc[[prediction_date]]

    # Reorder columns to match training order
    exog_for_rice_prediction = exog_for_rice_prediction[[
        'Teamperature', 'Inflation',
        'inflation_lag_1', 'inflation_lag_2', 'inflation_lag_3',
        'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3'
    ]]

    # 9. Predict rice price
    rice_prediction = loaded_rice_model.predict(
        start=len(loaded_rice_model.model.endog),
        end=len(loaded_rice_model.model.endog),
        exog=exog_for_rice_prediction
    )

    # 10. Display result
    if len(rice_prediction) > 0:
        predicted_price = rice_prediction.iloc[0]
        st.success(f"The predicted price of rice for {prediction_date.strftime('%B %Y')} is: ₦{predicted_price:.2f}")
    else:
        st.warning("Could not generate rice price prediction.")

