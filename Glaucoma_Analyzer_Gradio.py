import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def load_model(model_path):
    # Load your pre-trained model
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

# Replace this path with the path to your actual model
MODEL_PATH = "C:/AIProgram/Skripsi/Model_89.h5"
model = load_model(MODEL_PATH)

def diagnose_image(image):
    # Process the uploaded image and predict
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize the image to match the model's input size
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    glaucoma_probability = prediction[0][0]
    result_text = f"Probability of glaucoma: {glaucoma_probability:.2%}"
    
    return result_text

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Glaucoma Detection App")
        gr.Markdown("Upload an eye image to detect the probability of glaucoma.")
        
        with gr.Row():
            image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Diagnose")
        result = gr.Textbox(label="Diagnosis Result")
        
        image.change(fn=diagnose_image, inputs=image, outputs=result)
        submit_btn.click(fn=diagnose_image, inputs=image, outputs=result)
        
        gr.Markdown("### Glaucoma Analyzer V.1.0.0 by Thariq Arian")
        
    demo.launch()

if __name__ == "__main__":
    main()
