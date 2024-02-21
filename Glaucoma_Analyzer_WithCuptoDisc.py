import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

class GlaucomaDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Glaucoma Detection App")
        self.master.geometry("800x600")  # Set the initial size

        # Create buttons
        self.insert_button = tk.Button(master, text="Insert Image", command=self.open_image)
        self.insert_button.pack(pady=10)

        self.calculate_button = tk.Button(master, text="Calculate Cup-to-Disc Ratio", command=self.calculate_cup_disc_ratio)
        self.calculate_button.pack(pady=10)
        self.calculate_button["state"] = "disabled"  # Disable initially

        self.diagnose_button = tk.Button(master, text="Diagnose with CNN", command=self.diagnose_with_cnn)
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

        # Create a label and slider for threshold adjustment
        self.threshold_label = tk.Label(master, text="Threshold Value:")
        self.threshold_label.pack(pady=5)

        self.threshold_slider = tk.Scale(master, from_=0, to=255, orient="horizontal", command=self.update_threshold)
        self.threshold_slider.set(178)  # Set an initial value
        self.threshold_slider.pack(pady=5)

        # Create a label to display current threshold value
        self.current_threshold_label = tk.Label(master, text=f"Current Threshold: {self.threshold_slider.get()}")
        self.current_threshold_label.pack(pady=5)

        # Initialize variables
        self.image_path = None
        self.model_path = "C:/AIProgram/Model/Final_97CNN.h5"  # Specify the path to your model file
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        # Add window controls
        self.add_window_controls()

        # Add watermark
        self.add_watermark()

    def update_threshold(self, event):
        # Update the current threshold label when the slider is moved
        self.current_threshold_label.config(text=f"Current Threshold: {self.threshold_slider.get()}")

    def calculate_cup_disc_ratio(self):
        if self.image_path:
            # Load glaucoma image
            glaucoma_image = cv2.imread(self.image_path)

            # Perform optic disc and cup segmentation with adjustable threshold
            threshold_value = int(self.threshold_slider.get())
            disc_mask, cup_mask, segmented_disc, segmented_cup, disc_area, cup_area, cup_to_disc_ratio = segment_optic_disc_and_cup(glaucoma_image, threshold_value)

            # Display the images and cup-to-disc ratio
            self.display_segmentation_results(disc_mask, cup_mask, segmented_disc, segmented_cup)
            ratio_text = f"Cup-to-Disc Ratio: {cup_to_disc_ratio:.2f}"
            self.prediction_label.config(text=self.prediction_label.cget("text") + f"\n{ratio_text}")

    def diagnose_with_cnn(self):
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

            # Process and display results
            noglaucoma_probability = prediction[0][0]
            glaucoma_probability = 1 - noglaucoma_probability
            result_text = f"Probability of glaucoma: {glaucoma_probability:.2%}"
            self.prediction_label.config(text=result_text)

            # Enable diagnose button after processing
            self.diagnose_button["state"] = "active"
            self.file_label.config(text=f"Diagnosis completed for: {self.image_path}")

    def display_segmentation_results(self, disc_mask, cup_mask, segmented_disc, segmented_cup):
        plt.figure(figsize=(12, 4))

        plt.subplot(231), plt.imshow(disc_mask, cmap='gray'), plt.title('Optic Disc Binary Mask')
        plt.subplot(232), plt.imshow(cv2.cvtColor(segmented_disc, cv2.COLOR_BGR2RGB)), plt.title('Segmented Optic Disc')
        plt.subplot(233), plt.imshow(cup_mask, cmap='gray'), plt.title('Optic Cup Binary Mask')
        plt.subplot(234), plt.imshow(cv2.cvtColor(segmented_cup, cv2.COLOR_BGR2RGB)), plt.title('Segmented Optic Cup')

        plt.tight_layout()
        plt.show()

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
            self.calculate_button["state"] = "active"
            self.diagnose_button["state"] = "active"

    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((400, 400))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Keep a reference to the PhotoImage object
        self.canvas.image = self.photo

def segment_optic_disc_and_cup(glaucoma_image, threshold_value):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(glaucoma_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Use adaptive thresholding to segment the optic disc
    _, disc_mask = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Use adaptive thresholding to segment the optic cup
    _, cup_mask = cv2.threshold(blurred_image, threshold_value + 35, 255, cv2.THRESH_BINARY)

    # Find contours in the binary images
    disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank masks to draw the contours
    disc_contour_mask = np.zeros_like(gray_image)
    cup_contour_mask = np.zeros_like(gray_image)

    # Draw the contours on the masks
    cv2.drawContours(disc_contour_mask, disc_contours, -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(cup_contour_mask, cup_contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND the original image with the masks
    segmented_disc = cv2.bitwise_and(glaucoma_image, glaucoma_image, mask=disc_contour_mask)
    segmented_cup = cv2.bitwise_and(glaucoma_image, glaucoma_image, mask=cup_contour_mask)

    # Calculate cup-to-disc ratio
    disc_area = cv2.countNonZero(disc_mask)
    cup_area = cv2.countNonZero(cup_mask)
    cup_to_disc_ratio = cup_area / disc_area if disc_area != 0 else float('inf')

    return disc_mask, cup_mask, segmented_disc, segmented_cup, disc_area, cup_area, cup_to_disc_ratio

if __name__ == "__main__":
    root = tk.Tk()
    app = GlaucomaDetectionApp(root)
    root.mainloop()
