import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('D:\\Teachnook\\Project\\Code\\Models\\VGGdemo_trained_model.h5')

def preprocess_image(img):
    img = img.resize((224, 224))  # VGG16 input size
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)
    return img

st.title("VGG16 Clothing Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        st.write("Classifying...")
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        class_labels = ['Caps', 'Pants', 'Shoes', 'Shorts', 'tshirt']
        st.write(f"Predicted class: {class_labels[np.argmax(prediction)]}")

