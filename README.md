# dataset
# Image Colorization using Deep Learning- using pretrained models 
# Overview
This repository contains Python code for colorizing grayscale images using a pre-trained deep learning model. The model is implemented using the OpenCV library and utilizes a Convolutional Neural Network (CNN) with a Caffe model for colorization.

# Requirements
Python 3.x

OpenCV (cv2)

NumPy
# Model Files
colorization_deploy_v2.prototxt: Prototxt file specifying the architecture of the colorization model.
pts_in_hull.npy: NumPy file containing centers for ab channel quantization used for rebalancing.
colorization_release_v2.caffemodel: Pre-trained Caffe model file.
# Usage
Setup:

Ensure you have the required Python libraries installed: cv2, numpy.
Update the DIR variable in the code with the correct path to the model files.
Loading the Model:

The colorization model is loaded using the readNetFromCaffe function from OpenCV.
Configuration:

The model is configured with the necessary parameters, including points for ab channel quantization.
# Image Colorization:

Provide the path to the input grayscale image (1.jpg in this example) using the ImagePath variable.
The image is loaded, preprocessed, and the colorization model is applied.
# Display Results:
![image](https://github.com/phoenixERic/dataset/assets/137150955/f27bc5f2-7f37-4348-8488-bca88804cd9b)


The original and colorized images are displayed using OpenCV's imshow function.
Press any key to close the image windows.
# Notes
The model is designed to colorize grayscale images using the LAB color space.
The code includes necessary scaling and preprocessing steps.
# Image Colorization with Convolutional Neural Networks (CNNs)
This repository contains Python code for automatic image colorization using Convolutional Neural Networks (CNNs). The code utilizes the Keras library with a TensorFlow backend for building and training the model.

# Requirements
Python 3.x

NumPy

Pandas

OpenCV (cv2)

Matplotlib

Scikit-learn (sklearn)

Keras

TensorFlow

 # Dataset 
 kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving
![image](https://github.com/phoenixERic/dataset/assets/137150955/3adf72c7-f83f-47d7-8827-f3813afff08a)


The dataset used for training and testing the model is located in the dataset_updated directory. Make sure the dataset is structured appropriately for the code to load images successfully.

# Model Architecture
The model architecture consists of a Convolutional Neural Network (CNN) designed for image colorization. It uses a U-Net-like architecture with convolutional and upsampling layers. The model is compiled using the Adam optimizer with a specified learning rate and employs the mean squared error as the loss function.

# Training
The training process is configured with a generator function (GenerateInputs) to dynamically load and preprocess data during training. The model is trained for 53 epochs with a batch size of 1, optimizing for accuracy. Training history is visualized using Matplotlib, showing accuracy and loss trends over epochs.

# Usage
Ensure all required libraries are installed by running:

Execute the code in a Python environment, ensuring that the dataset is correctly structured.

# Results
![image](https://github.com/phoenixERic/dataset/assets/137150955/9fdb5887-cd87-4a1c-abf4-63c6b949ab6d)

The training process generates visualizations of accuracy and loss trends over epochs, providing insights into the model's performance. Adjust hyperparameters as needed for optimal results.



# License
This code is provided under the MIT License.
