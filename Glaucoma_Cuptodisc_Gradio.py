import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load your pre-trained model
model_path = "C:/AIProgram/Model/Final_97CNN.h5"  # Adjust path as necessary
model = tf.keras.models.load_model(model_path)

def segment_optic_disc_and_cup(image, threshold_value):
    """
    Function to segment the optic disc and cup from an eye image and calculate the cup-to-disc ratio.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Thresholding to segment the disc and cup
    _, disc_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
    _, cup_mask = cv2.threshold(blurred_image, threshold_value + 35, 255, cv2.THRESH_BINARY)

    # Find contours for disc and cup
    disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas
    disc_area = cv2.countNonZero(disc_mask)
    cup_area = cv2.countNonZero(cup_mask)
    cup_to_disc_ratio = cup_area / disc_area if disc_area != 0 else 0

    return cup_to_disc_ratio

def diagnose_image(image, threshold_value):
    """
    Process the uploaded image to predict glaucoma probability and calculate the cup-to-disc ratio.
    """
    img = np.array(image)  # Convert PIL Image to numpy array
    img_for_cnn = cv2.resize(img, (224, 224)) / 255.0  # Resize and normalize image for CNN
    img_for_cnn = np.expand_dims(img_for_cnn, axis=0)  # Add batch dimension

    # Make prediction with CNN
    prediction = model.predict(img_for_cnn)
    glaucoma_probability = 1 - prediction[0][0]  # Assuming binary classification
    
    # Calculate cup-to-disc ratio
    cup_to_disc_ratio = segment_optic_disc_and_cup(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), threshold_value)
    
    return f"Probability of Glaucoma: {glaucoma_probability:.2%}", f"Cup-to-Disc Ratio: {cup_to_disc_ratio:.2f}"

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Glaucoma Detection App")
        gr.Markdown("Upload an eye image to detect the probability of glaucoma and calculate the cup-to-disc ratio.")
        
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            threshold_slider = gr.Slider(minimum=0, maximum=255, value=178, label="Threshold Value")
        submit_button = gr.Button("Diagnose")
        
        result_text, cup_disc_ratio_text = gr.Textbox(label="Diagnosis Result"), gr.Textbox(label="Cup-to-Disc Ratio Result")
        
        submit_button.click(fn=diagnose_image, inputs=[image_input, threshold_slider], outputs=[result_text, cup_disc_ratio_text])
        
        gr.Markdown("### Glaucoma Analyzer V.1.0.0 by Thariq Arian")

    demo.launch()

if __name__ == "__main__":
    main()
