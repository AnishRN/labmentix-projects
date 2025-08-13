import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load your trained model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("VGG16_fish_classifier.h5")  # update with your file name
    return model

model = load_model()

# --- Class labels ---
CLASS_NAMES = [
    'Sea Bass',
    'Horse Mackerel',
    'Shrimp',
    'Striped Red Mullet',
    'Black Sea Sprat',
    'Gilt-Head Bream',
    'Red Mullet',
    'Fish',
    'Red Sea Bream',
    'Trout',
    'Bass'
]

# --- Preprocessing (MATCHES your model input shape) ---
def preprocess_image(img):
    target_size = (128, 128)  # model input size
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit UI ---
st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟")
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and get the predicted species with confidence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
