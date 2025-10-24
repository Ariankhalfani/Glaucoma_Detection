markdown# Glaucoma Detection App

A desktop application for automated glaucoma detection using deep learning and image processing techniques. This application provides both traditional cup-to-disc ratio calculation and CNN-based diagnosis for glaucoma screening from retinal fundus images.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Image Upload**: Support for PNG, JPG, and JPEG retinal fundus images
- **Cup-to-Disc Ratio Calculation**: Automated segmentation of optic disc and optic cup with adjustable threshold
- **CNN-Based Diagnosis**: Deep learning model for glaucoma probability prediction
- **Interactive Threshold Control**: Real-time adjustment of segmentation parameters
- **Visual Results**: Display of segmentation masks and processed images
- **User-Friendly GUI**: Built with Tkinter for easy interaction

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Information](#model-information)
- [Technical Details](#technical-details)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/glaucoma-detection-app.git
cd glaucoma-detection-app
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the Model

Place your trained CNN model file (`Final_97CNN.h5`) in the appropriate directory:
```
C:/AIProgram/Model/Final_97CNN.h5
```

Or modify the `model_path` variable in the code to point to your model location.

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.4.0
Pillow>=8.3.0
```

### System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 500MB free space for model and dependencies

## Usage

### Running the Application
```bash
python glaucoma_detection_app.py
```

### Step-by-Step Guide

1. **Launch the Application**
   - Run the Python script to open the GUI window

2. **Insert Image**
   - Click the "Insert Image" button
   - Select a retinal fundus image (PNG, JPG, or JPEG format)
   - The image will be displayed in the canvas

3. **Calculate Cup-to-Disc Ratio**
   - Adjust the threshold slider (0-255) to optimize segmentation
   - Click "Calculate Cup-to-Disc Ratio"
   - View the segmentation results in separate windows
   - The cup-to-disc ratio will be displayed below the buttons

4. **CNN Diagnosis**
   - Click "Diagnose with CNN" to get AI-powered prediction
   - Wait for the diagnosis to complete
   - View the probability of glaucoma in the results area

### Interpreting Results

#### Cup-to-Disc Ratio
- **Normal**: < 0.3
- **Borderline**: 0.3 - 0.6
- **Glaucoma Suspect**: > 0.6

#### CNN Diagnosis
- The model outputs a probability percentage for glaucoma presence
- Higher percentages indicate greater likelihood of glaucoma
- Results should be verified by medical professionals

## Model Information

### CNN Architecture

The application uses a pre-trained Convolutional Neural Network model:

- **Input Size**: 224x224x3 (RGB images)
- **Model File**: `Final_97CNN.h5`
- **Reported Accuracy**: 97% (as indicated by filename)
- **Output**: Binary classification (Glaucoma/No Glaucoma)

### Image Preprocessing

1. Resize to 224x224 pixels
2. Normalize pixel values to [0, 1] range
3. Add batch dimension for model input

## Technical Details

### Image Segmentation Algorithm

The application uses computer vision techniques for optic disc and cup segmentation:

1. **Grayscale Conversion**: Convert RGB image to grayscale
2. **Gaussian Blur**: Apply 7x7 Gaussian filter to reduce noise
3. **Thresholding**: Use adjustable threshold for binary segmentation
   - Optic Disc: User-defined threshold value
   - Optic Cup: Threshold value + 35
4. **Contour Detection**: Find and fill contours using OpenCV
5. **Ratio Calculation**: Cup area / Disc area

### Key Functions

#### `segment_optic_disc_and_cup(glaucoma_image, threshold_value)`
Performs segmentation of optic disc and cup regions.

**Parameters:**
- `glaucoma_image`: Input BGR image
- `threshold_value`: Threshold for binary segmentation (0-255)

**Returns:**
- `disc_mask`: Binary mask of optic disc
- `cup_mask`: Binary mask of optic cup
- `segmented_disc`: Color image of segmented disc
- `segmented_cup`: Color image of segmented cup
- `disc_area`: Pixel count of disc area
- `cup_area`: Pixel count of cup area
- `cup_to_disc_ratio`: Calculated ratio

## Screenshots

*Add screenshots of your application here:*

### Main Interface
```
[Screenshot of main window with image loaded]
```

### Segmentation Results
```
[Screenshot of segmentation visualization]
```

### Diagnosis Results
```
[Screenshot showing CNN prediction]
```

## Project Structure
```
glaucoma-detection-app/
│
├── glaucoma_detection_app.py    # Main application file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # License file
│
├── Model/
│   └── Final_97CNN.h5           # Trained CNN model
│
├── docs/
│   └── user_guide.md            # Detailed user guide
│
└── examples/
    ├── sample_image1.jpg        # Sample retinal images
    └── sample_image2.jpg
```

## Limitations

- **Not for Clinical Diagnosis**: This application is for educational and research purposes only
- **Image Quality**: Results depend on image quality and proper fundus photography
- **Threshold Sensitivity**: Segmentation results vary with threshold settings
- **Model Limitations**: The CNN model may not generalize to all populations or image types

## Future Enhancements

- [ ] Support for batch processing of multiple images
- [ ] Export results to PDF report
- [ ] Integration with DICOM format
- [ ] Real-time camera capture support
- [ ] Multi-language support
- [ ] Cloud-based model deployment
- [ ] Advanced visualization with heatmaps
- [ ] Database integration for patient records

## Troubleshooting

### Common Issues

**Issue**: Model file not found
```
Solution: Verify the model path in line 49 of the code matches your actual model location
```

**Issue**: ImportError for TensorFlow
```
Solution: Install TensorFlow with: pip install tensorflow
```

**Issue**: Segmentation results are poor
```
Solution: Adjust the threshold slider to optimize segmentation for your specific image
```

**Issue**: Application crashes on diagnosis
```
Solution: Ensure your model file is compatible with the installed TensorFlow version
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation for any changes
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**IMPORTANT**: This application is intended for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for proper glaucoma screening and diagnosis.

## Citation

If you use this application in your research, please cite:
```bibtex
@software{glaucoma_detection_app,
  author = {Thariq Arian},
  title = {Glaucoma Detection App},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/yourusername/glaucoma-detection-app}
}
```

## Author

**Thariq Arian**

- GitHub: [@ariankhalfani](https://github.com/ariankhalfani)
- Email: ariankhalfani@gmail.com

## Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Medical professionals who provided domain expertise
- Contributors and testers


---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Status**: Active Development
