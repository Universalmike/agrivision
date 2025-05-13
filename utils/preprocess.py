import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Preprocess image before feeding to model (resize and normalize)
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to fit model input
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
