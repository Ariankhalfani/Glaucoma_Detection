import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPooling2D

class GlaucomaDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Glaucoma Detection App")
        self.master.geometry("800x600")  # Set the initial size

        # Create buttons
        self.insert_button = tk.Button(master, text="Insert Image", command=self.open_image)
        self.insert_button.pack(pady=10)

        self.diagnose_button = tk.Button(master, text="Diagnose", command=self.diagnose_image)
        self.diagnose_button.pack(pady=10)
        self.diagnose_button["state"] = "disabled"  # Disable initially

        # Create label to display inserted image file path and model status
        self.file_label = tk.Label(master, text="")
        self.file_label.pack(pady=10)

        # Create label to display prediction result
        self.prediction_label = tk.Label(master, text="")
        self.prediction_label.pack(pady=10)

        # Create image canvas
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()

        # Initialize variables
        self.image_path = None
        self.model_path = "C:/AIProgram/final_85.h5"  # Specify the path to your model file
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        # Add window controls
        self.add_window_controls()

        # Add watermark
        self.add_watermark()

    def add_window_controls(self):
        # Close button
        close_button = tk.Button(self.master, text="Close", command=self.master.destroy)
        close_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Minimize button
        minimize_button = tk.Button(self.master, text="Minimize", command=self.master.iconify)
        minimize_button.pack(side=tk.RIGHT, pady=10)

    def add_watermark(self):
        watermark_label = tk.Label(self.master, text="Glaucoma Analyzer V.1.0.0 by Thariq Arian", font=("Arial", 8), fg="gray")
        watermark_label.place(relx=1.0, rely=1.0, anchor=tk.SE)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if self.image_path:
            self.file_label.config(text=f"Inserted Image: {self.image_path}")
            self.display_image()
            self.diagnose_button["state"] = "active"

    def diagnose_image(self):
        if self.image_path and self.model:
            # Load and preprocess the image
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (224, 224))  # Resize the image to match the model's input size
            img = img / 255.0  # Normalize the pixel values to be between 0 and 1
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Disable diagnose button during processing
            self.diagnose_button["state"] = "disabled"

            # Show buffering effect
            self.file_label.config(text="Diagnosing. Please wait...")
            self.master.update()

            # Make prediction
            prediction = self.model.predict(img)

            # Process and display results (customize as needed)
            noglaucoma_probability = prediction[0][0]
            glaucoma_probability = 1 - noglaucoma_probability
            result_text = f"Probability of glaucoma: {glaucoma_probability:.2%}"
            self.prediction_label.config(text=result_text)

            # Enable diagnose button after processing
            self.diagnose_button["state"] = "active"
            self.file_label.config(text=f"Diagnosis completed for: {self.image_path}")

    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((400, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Keep a reference to the PhotoImage object
        self.canvas.image = self.photo


if __name__ == "__main__":
    root = tk.Tk()
    app = GlaucomaDetectionApp(root)
    root.mainloop()
