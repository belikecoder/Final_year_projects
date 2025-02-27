import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import os

# Load the trained model
model = load_model('lung_cancer_detection_model.h5')

# Define categories
categories = ['Benign cases', 'Malignant cases', 'Normal cases']

# Streamlit page configuration
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

# Apply custom CSS for better UI
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🩺 Lung Cancer Detection System")
st.write("Upload a lung scan image to predict whether it's benign, malignant, or normal.")

# Function to preprocess the uploaded image
def preprocess_image(img, is_grayscale=True):
    img = img.resize((256, 256))  # Resize image
    if is_grayscale:
        img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], help="Upload an X-ray or CT scan image.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼️ Uploaded Image', use_column_width=True)
    is_grayscale = st.checkbox('Is this image grayscale?', value=False)
    
    # Preprocess the image
    img_array = preprocess_image(img, is_grayscale)
    
    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner('Analyzing image...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = categories[predicted_class]
            
            # Display results
            st.success(f"✅ Predicted Class: {predicted_label}")
            st.write("### 🏆 Prediction Probabilities:")
            for i, category in enumerate(categories):
                st.progress(int(predictions[0][i] * 100))
                st.write(f"**{category}:** {predictions[0][i] * 100:.2f}%")
