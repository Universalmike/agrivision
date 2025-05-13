import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import calendar
import io
import pickle
from tensorflow.keras.preprocessing import image
from langchain.vectorstores import Chroma
from utils.preprocess import preprocess_image
from utils.translate import translate_answer
from FARMAN.loader import load_and_split
from FARMAN.vector_sector import create_vector_store
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Load ARIMA models
with open("models/rice_model.pkl", "rb") as f:
    rice_model = pickle.load(f)

with open("models/maize_model.pkl", "rb") as f:
    maize_model = pickle.load(f)

# Load TensorFlow image classifier model
model = load_model("plant_disease_classifier.h5")

# ARIMA prediction function
def predict_price(crop, dates):
    if crop == "Rice":
        model = rice_model
    else:
        model = maize_model

    forecast_steps = 24
    forecast = model.forecast(steps=forecast_steps)

    base_year = 2023
    base_month = 12

    print(f"{crop} forecast length: {len(forecast)}")

    predictions = []
    for d in dates:
        months_since_base = (d.year - base_year) * 12 + (d.month - base_month)
        if 0 <= months_since_base < len(forecast):
            try:
                predictions.append(forecast.iloc[months_since_base])
            except Exception as e:
                print(f"Error for {d}: {e}")
                predictions.append(np.nan)
        else:
            predictions.append(np.nan)

    return pd.DataFrame({"Date": dates, "Predicted Price": predictions})



# Leaf classifier function

class_labels = [
    'Tomato__Tomato_mosaic_virus', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato_Leaf_Mold',
    'Potato___healthy', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot',
    'Tomato_healthy', 'Tomato_Septoria_leaf_spot', 'Pepper__bell___healthy',
    'Pepper__bell___Bacterial_spot', 'Potato___Early_blight', 'Tomato_Late_blight',
    'Tomato_Early_blight'
]

def classify_leaf(image_file):
    # Ensure image_file is a stream
    image_bytes = image_file.read() if hasattr(image_file, 'read') else image_file.getvalue()
    img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    predicted_probs = model.predict(img_array)
    predicted_index = np.argmax(predicted_probs)
    predicted_label = class_labels[predicted_index]
    predicted_label = class_labels[np.argmax(predicted_probs)]
    confidence = predicted_probs[predicted_index]
    return f"{predicted_label} (Confidence: {confidence:.2f})"


st.title("Agri Forecast & Leaf Classifier App")

# Tabs for sections
tabs = st.tabs(["Price Prediction", "Leaf Image Classification"])

# --- Section 1: Price Prediction ---
with tabs[0]:
    st.header("Rice and Maize Price Prediction")

    crop_choice = st.radio("Choose Crop", ["Rice", "Maize"])

    # Input multiple dates with minimum constraint
    selected_dates = st.date_input(
        "Select date",
        [],
        min_value=datetime(2024, 1, 1),
        format="YYYY-MM-DD"
    )

    # Adjust day to end of the month
    adjusted_dates = sorted(set(
        datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
        for d in selected_dates
    ))

    if adjusted_dates:
        df = predict_price(crop_choice, adjusted_dates)
        st.write("### Price Prediction Table")
        st.dataframe(df)

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted Price"], mode='lines+markers', name=crop_choice))
        fig.update_layout(title=f"{crop_choice} Price Trend", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

# --- Section 2: Leaf Image Classification ---
with tabs[1]:
    st.header("Leaf Image Health Classifier")

    uploaded_files = st.file_uploader("Upload leaf images", type=["jpg", "png"], accept_multiple_files=True)
    camera_image = st.camera_input("Or take a picture")

    images = []
    if uploaded_files:
        images.extend(uploaded_files)
    elif camera_image:
        images.append(camera_image)

    if images:
        for img in images:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            #img_bytes = img.read() if hasattr(img, 'read') else img.getvalue()
            with st.spinner("Classifying..."):
                result = classify_leaf(img)
                st.success(f"Prediction: {result}")

    st.subheader("Classify Leaves in Video")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        st.write("Extracting and classifying frames...")
        frame_count = 0
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame).resize((224, 224))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            result = classify_leaf(buf.getvalue())
            st.image(pil_img, caption=f"Frame {frame_count+1} - {result}")
            frame_count += 5
        cap.release()
