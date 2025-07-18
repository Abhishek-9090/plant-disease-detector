import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model("plant_disease_model.h5")

# Class labels
class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple__Cedar_apple_rust',
               'Apple__healthy', 'Blueberry_healthy', 'Cherry__Powdery_mildew',
               'Cherry__healthy', 'Corn__Cercospora_leaf_spot Gray_leaf_spot',
               'Corn__Common_rust', 'Corn_healthy', 'Grape__Black_rot',
               'Grape__Esca(Black_Measles)', 'Grape___healthy',
               'Potato__Early_blight', 'Potato__Late_blight']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))  # ⚠ Use (128,128) as per model
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)
    return image

# Medicine suggestion dictionary
medicine_dict = {
    'Apple___Apple_scab': 'Use fungicide containing Captan.',
    'Apple___Black_rot': 'Remove infected fruits. Spray Thiophanate-methyl.',
    'Apple___Cedar_apple_rust': 'Use Mancozeb fungicide before rainy season.',
    'Apple___healthy': 'No disease detected.',
    'Blueberry___healthy': 'No disease detected.',
    'Cherry___Powdery_mildew': 'Apply Sulfur-based fungicide.',
    'Cherry___healthy': 'No disease detected.',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Spray with fungicide containing Pyraclostrobin.',
    'Corn___Common_rust': 'Use fungicide with Azoxystrobin.',
    'Corn___healthy': 'No disease detected.',
    'Grape___Black_rot': 'Spray Myclobutanil or Captan.',
    'Grape__Esca(Black_Measles)': 'No cure. Remove infected vines.',
    'Grape___healthy': 'No disease detected.',
    'Potato___Early_blight': 'Use Chlorothalonil based fungicide.',
    'Potato___Late_blight': 'Spray with Metalaxyl.',
}

# Streamlit UI
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a plant leaf image to detect disease and get medicine suggestion.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img_array = preprocess_image(image)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"🧬 Predicted: *{predicted_class}* ({confidence*100:.2f}% confidence)")

    # Suggest medicine
    if predicted_class in medicine_dict:
        st.info(f"💊 *Medicine Suggestion*: {medicine_dict[predicted_class]}")
    else:
        st.warning("No medicine suggestion found for this class.")