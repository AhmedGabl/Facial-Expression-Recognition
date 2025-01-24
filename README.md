# Facial Expression Recognition

## Description
This repository contains a deep learning solution for facial expression recognition using grayscale images of faces. The task is to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The dataset consists of 48x48 pixel grayscale images of faces, automatically aligned to occupy the same space in each image.

### Key Features:
- Uses TensorFlow and Keras for model building.
- Implements data augmentation and normalization to improve performance.
- Supports multi-class classification with a robust convolutional neural network (CNN).
- Includes visualization of training metrics.

## Dataset
The dataset used is FER-2013, which consists of:
- **Training set**: 28,709 images
- **Test set**: 3,589 images

### Categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following structure:
- **Input layer**: Accepts 48x48 grayscale images.
- **Convolutional layers**: Extracts features with filters of increasing depth.
- **Pooling layers**: Reduces spatial dimensions.
- **Dense layers**: Performs classification with ReLU and softmax activation.

## Results
The model achieves accurate recognition across all seven emotion categories. Metrics such as accuracy and loss are visualized during training.

## Acknowledgments
- The FER-2013 dataset creators.
- TensorFlow and Keras for their powerful deep learning tools.


