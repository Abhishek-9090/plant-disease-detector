import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model("plant_disease_model.h5")

# Class labels
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)_Powdery_mildew",
    "Cherry_(including_sour)_healthy",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)Common_rust",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Corn_(maize)_healthy",
    "Grape___Black_rot",
    "Grape__Esca(Black_Measles)",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)",
    "Orange___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,bell__Bacterial_spot",
    "Pepper,bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Squash___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Mango___Powdery_mildew",
    "Mango___Anthracnose",
    "Mango___Bacterial_canker",
    "Mango___healthy",
    "Wheat___Leaf_rust",
    "Wheat___Stem_rust",
    "Wheat___Loose_smut",
    "Wheat___healthy",
    "Brinjal___Phomopsis_blight",
    "Brinjal___Bacterial_wilt",
    "Brinjal___Little_leaf_disease",
    "Brinjal___healthy",
    "Rice___Blast",
    "Rice___Brown_spot",
    "Rice___Bacterial_leaf_blight",
    "Rice___healthy",
    "Banana___Panama_wilt",
    "Banana___Sigatoka",
    "Banana___healthy",
    "Pomegranate___Bacterial_blight",
    "Pomegranate___Leaf_spot",
    "Pomegranate___healthy",
    "Guava___Anthracnose",
    "Guava___Wilt",
    "Guava___healthy",
    "Papaya___Powdery_mildew",
    "Papaya___Leaf_curl",
    "Papaya___healthy",
    "Sugarcane___Red_rot",
    "Sugarcane___Smut",
    "Sugarcane___healthy",
    "Cotton___Leaf_curl_virus",
    "Cotton___Bacterial_blight",
    "Cotton___healthy",
    "Chili___Anthracnose",
    "Chili___Leaf_curl",
    "Chili___healthy",
    "Okra___Yellow_vein_mosaic",
    "Okra___Powdery_mildew",
    "Okra___healthy"
]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))  # ⚠ Use (128,128) as per model
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)
    return image

# Medicine suggestion dictionary
medicine_dict = {
    "Apple___Apple_scab": ["Mancozeb", "Captan", "Sulfur Spray"],
    "Apple___Black_rot": ["Carbendazim", "Copper-based fungicide"],
    "Apple___Cedar_apple_rust": ["Myclobutanil", "Propiconazole"],
    "Apple___healthy": ["No disease detected"],
    "Blueberry___healthy": ["No disease detected"],
    "Cherry_(including_sour)_Powdery_mildew": ["Neem Oil", "Sulfur Fungicide"],
    "Cherry_(including_sour)_healthy": ["No disease detected"],
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot": ["Propiconazole", "Azoxystrobin"],
    "Corn_(maize)Common_rust": ["Chlorothalonil", "Maneb"],
    "Corn_(maize)_Northern_Leaf_Blight": ["Pyraclostrobin", "Tebuconazole"],
    "Corn_(maize)_healthy": ["No disease detected"],
    "Grape___Black_rot": ["Myclobutanil", "Trifloxystrobin"],
    "Grape__Esca(Black_Measles)": ["Tebuconazole", "Trifloxystrobin"],
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)": ["Copper fungicide", "Mancozeb"],
    "Grape___healthy": ["No disease detected"],
    "Orange__Haunglongbing(Citrus_greening)": ["No cure. Remove infected trees", "Use insecticides for psyllid control"],
    "Orange___healthy": ["No disease detected"],
    "Peach___Bacterial_spot": ["Copper Spray", "Oxytetracycline"],
    "Peach___healthy": ["No disease detected"],
    "Pepper,bell__Bacterial_spot": ["Streptocycline", "Copper Oxychloride"],
    "Pepper,bell__healthy": ["No disease detected"],
    "Potato___Early_blight": ["Chlorothalonil", "Mancozeb"],
    "Potato___Late_blight": ["Metalaxyl", "Ridomil Gold"],
    "Potato___healthy": ["No disease detected"],
    "Raspberry___healthy": ["No disease detected"],
    "Soybean___healthy": ["No disease detected"],
    "Squash___Powdery_mildew": ["Sulfur", "Potassium bicarbonate"],
    "Squash___healthy": ["No disease detected"],
    "Strawberry___Leaf_scorch": ["Captan", "Myclobutanil"],
    "Strawberry___healthy": ["No disease detected"],
    "Tomato___Bacterial_spot": ["Copper Oxychloride", "Streptocycline"],
    "Tomato___Early_blight": ["Chlorothalonil", "Mancozeb"],
    "Tomato___Late_blight": ["Metalaxyl", "Mancozeb"],
    "Tomato___Leaf_Mold": ["Mancozeb", "Copper fungicide"],
    "Tomato___Septoria_leaf_spot": ["Chlorothalonil", "Neem oil"],
    "Tomato___Spider_mites Two-spotted_spider_mite": ["Miticides", "Insecticidal soap"],
    "Tomato___Target_Spot": ["Chlorothalonil", "Azoxystrobin"],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["Use resistant varieties", "Control whiteflies"],
    "Tomato___Tomato_mosaic_virus": ["Remove infected plants", "Disinfect tools"],
    "Tomato___healthy": ["No disease detected"],
    "Mango___Powdery_mildew": ["Sulfur dust", "Karathane"],
    "Mango___Anthracnose": ["Copper oxychloride", "Carbendazim"],
    "Mango___Bacterial_canker": ["Streptocycline", "Bordeaux mixture"],
    "Mango___healthy": ["No disease detected"],
    "Wheat___Leaf_rust": ["Propiconazole", "Mancozeb"],
    "Wheat___Stem_rust": ["Tebuconazole", "Triadimefon"],
    "Wheat___Loose_smut": ["Seed treatment with Carboxin", "Thiram"],
    "Wheat___healthy": ["No disease detected"],
    "Brinjal___Phomopsis_blight": ["Carbendazim", "Chlorothalonil"],
    "Brinjal___Bacterial_wilt": ["Soil drenching with bleaching powder", "Crop rotation"],
    "Brinjal___Little_leaf_disease": ["Control leafhoppers", "Rogor spray"],
    "Brinjal___healthy": ["No disease detected"],
    "Rice___Blast": ["Tricyclazole", "Isoprothiolane"],
    "Rice___Brown_spot": ["Mancozeb", "Copper oxychloride"],
    "Rice___Bacterial_leaf_blight": ["Streptocycline", "Copper Hydroxide"],
    "Rice___healthy": ["No disease detected"],
    "Banana___Panama_wilt": ["Carbendazim", "Soil solarization"],
    "Banana___Sigatoka": ["Propiconazole", "Neem Oil"],
    "Banana___healthy": ["No disease detected"],
    "Pomegranate___Bacterial_blight": ["Copper fungicide", "Streptocycline"],
    "Pomegranate___Leaf_spot": ["Mancozeb", "Neem extract"],
    "Pomegranate___healthy": ["No disease detected"],
    "Guava___Anthracnose": ["Carbendazim", "Chlorothalonil"],
    "Guava___Wilt": ["Soil drenching with Carbendazim", "Crop rotation"],
    "Guava___healthy": ["No disease detected"],
    "Papaya___Powdery_mildew": ["Sulfur spray", "Wettable sulfur"],
    "Papaya___Leaf_curl": ["Rogor", "Insecticidal soap"],
    "Papaya___healthy": ["No disease detected"],
    "Sugarcane___Red_rot": ["Carbendazim", "Seed treatment with Bavistin"],
    "Sugarcane___Smut": ["Thiram", "Hot water treatment"],
    "Sugarcane___healthy": ["No disease detected"],
    "Cotton___Leaf_curl_virus": ["Use resistant varieties", "Whitefly control"],
    "Cotton___Bacterial_blight": ["Copper oxychloride", "Streptocycline"],
    "Cotton___healthy": ["No disease detected"],
    "Chili___Anthracnose": ["Mancozeb", "Chlorothalonil"],
    "Chili___Leaf_curl": ["Dimethoate", "Rogor"],
    "Chili___healthy": ["No disease detected"],
    "Okra___Yellow_vein_mosaic": ["Imidacloprid", "Neem oil spray"],
    "Okra___Powdery_mildew": ["Sulfur", "Trifloxystrobin"],
    "Okra___healthy": ["No disease detected"]
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