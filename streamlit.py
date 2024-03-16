import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to diagnose the uploaded image
def diagnose_image(image, model):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    glaucoma_probability = prediction[0][0]
    return glaucoma_probability

def main():
    st.title("Glaucoma Detection App")
    st.markdown("Upload an eye image to detect the probability of glaucoma.")
    
    model_path = "C:/AIProgram/Skripsi/Model_89.h5"
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Diagnosing...")

        # Call diagnose_image function
        glaucoma_probability = diagnose_image(image, model)

        st.write(f"Probability of glaucoma: {glaucoma_probability:.2%}")

    st.markdown("### Glaucoma Analyzer V.1.0.0 by Thariq Arian")

if __name__ == "__main__":
    main()
