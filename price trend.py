import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from PIL import Image
import pickle
from langchain.vectorstores import Chroma
from utils.preprocess import preprocess_image
from utils.translate import translate_answer
from FARMAN.loader import load_and_split
from FARMAN.vector_sector import create_vector_store
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Load ARIMA models
with open("rice_model.pkl", "rb") as f:
    rice_model = pickle.load(f)

with open("maize_model.pkl", "rb") as f:
    maize_model = pickle.load(f)

# Load TensorFlow image classifier model
leaf_model = tf.keras.models.load_model("plant_disease_classifier.h5")

# ARIMA prediction function
def predict_price(crop, dates):
    model = rice_model if crop == "Rice" else maize_model
    start = (dates[0] - datetime(2023, 12, 31)).days
    end = (dates[-1] - datetime(2023, 12, 31)).days
    forecast = model.predict(start=start, end=end)
    predictions = [forecast[(d - datetime(2023, 12, 31)).days] for d in dates]
    return pd.DataFrame({"Date": dates, "Price": predictions})

# Leaf classifier function
def classify_leaf(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = leaf_model.predict(img_array)[0][0]
    return "Healthy" if prediction > 0.5 else "Spoilt"

st.title("Agri Forecast & Leaf Classifier App")

# Tabs for sections
tabs = st.tabs(["Price Prediction", "Leaf Image Classification"])

# --- Section 1: Price Prediction ---
with tabs[0]:
    st.header("Rice and Maize Price Prediction")

    crop_choice = st.radio("Choose Crop", ["Rice", "Maize"])

    # Input multiple dates with minimum constraint
    selected_dates = st.date_input(
        "Select future dates (after 2023-12-31):",
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
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Price"], mode='lines+markers', name=crop_choice))
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
            img_bytes = img.read() if hasattr(img, 'read') else img.getvalue()
            result = classify_leaf(img_bytes)
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


    
st.title("AI Post-Harvest Advisory Chatbot")

if "qa_chain" not in st.session_state:
    st.write("Initializing document intelligence...")
    docs = load_and_split("FARMAN/S-9-3-40-764.pdf")  # replace with your file
    db = create_vector_store(docs)

query = st.text_input("Ask a question about storage, pests, or preservation")

if query:
    retriever = db.as_retriever(search_kwargs={"k": 15})  # Retrieve top 3 most relevant chunks
    #query = "What are the emergimg technologies fr pest control"
    retrieved_docs = retriever.get_relevant_documents(query)
    response = generate_response(query, retrieved_docs)
    st.write("### Answer:")
    st.write(response)
