import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from PIL import Image
from utils.preprocess import preprocess_image
from utils.translate import translate_answer
from FARMAN.loader import load_and_split
from FARMAN.vector_sector import create_vector_store

# Load ARIMA Model (assuming it's saved as a pickle)
@st.cache_resource
def load_arima_model():
    # Load your trained ARIMA model here (replace with actual path)
    import pickle
    with open('rice_price_arimax_model.pkl', 'rb') as file:
        arima_model = pickle.load(file)
    return arima_model

# Load plant classifier model (TensorFlow)
@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model('plant_disease_classifier.h5')

# ARIMA Prediction Logic
def predict_rice_price(model, month, temperature):
    # Example: Predict for next month (based on ARIMA trained with inflation, temperature)
    # You might need to adjust this depending on how your ARIMA was trained
    forecast = model.predict(start=len(month) + 1, end=len(month) + 1)
    predicted_price = forecast[0] * (1 + temperature / 100)  # Adjust for temperature influence
    return predicted_price

# Temperature slider logic
def get_temperature_slider(df):
    mean_temp = df['temperature'].mean()
    return st.slider("Select Temperature for Prediction", min_value=int(mean_temp - 2), max_value=int(mean_temp + 2), value=int(mean_temp))

# Load data (Simulate reading from CSV)
df = pd.read_csv('Market Price data - 2007 to 2023.csv', parse_dates=['Date'], index_col='Date')
model = load_arima_model()
plant_model = load_plant_model()

# Sidebar UI elements
st.sidebar.title("Rice Price Prediction & Plant Health")
month_input = st.sidebar.date_input("Select the Month to Predict", value=pd.to_datetime("2023-12-01"))
temperature_input = get_temperature_slider(df)

# Image Classification for Plant Leaf (TensorFlow)
uploaded_image = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify Leaf Health"):
        input_tensor = preprocess_image(image)
        predictions = plant_model.predict(input_tensor)
        predicted_class = np.argmax(predictions, axis=1)
        class_names = ['Healthy', 'Diseased']  # Adjust based on your model
        st.success(f"Prediction: **{class_names[predicted_class[0]]}**")
        st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")

# Rice price prediction
if month_input:
    predicted_price = predict_rice_price(model, df, temperature_input)
    st.write(f"Predicted Rice Price for {month_input.strftime('%B %Y')}: {predicted_price:.2f} NGN")
    
st.title("AI Post-Harvest Advisory Chatbot")

if "qa_chain" not in st.session_state:
    st.write("Initializing document intelligence...")
    docs = load_and_split("data/tomato_storage_fao.pdf")  # replace with your file
    db = create_vector_store(docs)

query = st.text_input("Ask a question about storage, pests, or preservation")

if query:
    retriever = db.as_retriever(search_kwargs={"k": 15})  # Retrieve top 3 most relevant chunks
    #query = "What are the emergimg technologies fr pest control"
    retrieved_docs = retriever.get_relevant_documents(query)
    response = generate_response(query, retrieved_docs)
    st.write("### Answer:")
    st.write(response)
