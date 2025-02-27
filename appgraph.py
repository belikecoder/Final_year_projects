import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

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
        .stTextInput, .stFileUploader {
            border-radius: 10px;
        }
        .stProgress > div > div > div > div {
            background-color: #ff4b4b;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🩺 Lung Cancer Detection System")
st.markdown(
    "### Upload a lung scan image to predict whether it's benign, malignant, or normal."
)

# Function to preprocess the uploaded image
def preprocess_image(img, is_grayscale=True):
    img = img.resize((256, 256))  # Resize image
    
    if is_grayscale or model.input_shape[-1] == 1:  
        img = img.convert('L')  # Convert to grayscale (1 channel)
        img_array = np.array(img).reshape(256, 256, 1)  # Reshape for single channel
    else:
        img_array = np.array(img).reshape(256, 256, 3)  # Keep 3 channels if needed

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"], help="Upload an X-ray or CT scan image.")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼️ Uploaded Image', use_column_width=True)
    is_grayscale = st.checkbox('Is this image grayscale?', value=False)
    
    # Preprocess the image
    img_array = preprocess_image(img, is_grayscale)
    
    # Predict button
    if st.button("🔍 Predict", key="predict_button"):
        with st.spinner('⏳ Analyzing image...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = categories[predicted_class]
            
            # Display results
            st.success(f"✅ **Predicted Class:** {predicted_label}")
            st.markdown("### 🏆 Prediction Probabilities:")
            
            # Create probability dataframe
            prob_data = {
                "Category": categories,
                "Probability": predictions[0]
            }
            prob_df = pd.DataFrame(prob_data)
            
            # Display as bar chart
            st.bar_chart(prob_df.set_index('Category')['Probability'])
            
            # Display as table
            st.table(prob_df)
