# Pothole Detection System

## Overview
The Pothole Detection System is a convolutional neural network (CNN) designed to accurately identify potholes in real-time using dashcam footage. The model balances complexity and efficiency, ensuring detection accuracy while being capable of running on typical onboard hardware.

## Features
- **Real-time pothole detection** using dashcam footage
- Optimized for efficiency and accuracy
- Designed to run on standard onboard hardware
- Preprocessing and normalization of input images
- Multiple convolutional layers for feature extraction
- Max-pooling layers for downsampling and computational efficiency
- Dropout layers to prevent overfitting
- Fully connected layers for final classification

## Model Architecture
1. **Input Layer**
   - Accepts resized images (e.g., 64x64 pixels)
   - Normalizes pixel values to enhance performance

2. **Convolutional Layers**
   - Extracts features such as edges, textures, and shapes
   - Utilizes ReLU activation functions for non-linearity

3. **Pooling Layers**
   - Downsamples feature maps using max-pooling
   - Captures dominant features, improving model invariance to translations and distortions

4. **Dropout Layers**
   - Prevents overfitting by randomly setting a fraction of input units to zero during training
   - Encourages learning of robust features

5. **Fully Connected Layers**
   - Aggregates features extracted by convolutional layers
   - Performs final classification based on learned features

6. **Output Layer**
   - Uses softmax activation to provide class probabilities (pothole or normal road)
   - Selects the class with the highest probability as the prediction

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/padhisneha/potholedetectionusingcnn.git

2. Navigate to the project directory:
   ```bash
   cd potholedetectionusingcnn

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## Usage
1. Prepare your dataset with dashcam footage images.
2. Preprocess the images (resizing and normalization).
3. Train the model: Open 'sneha-anushka.ipynb' in Jupyter Notebook and run the cells to train the model.
4. Use the trained model for real-time detection:
   ```bash
   python project.py
   python gui.py


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.


## Contact
For any questions or suggestions, feel free to open an issue or contact me at sneha.padhi0408@gmail.com.

