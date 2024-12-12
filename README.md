# Image-Classification-using-CNN


This repository contains the implementation of a simple Convolutional Neural Network (CNN) for image classification, created as part of **EN3150 Assignment 03** in Pattern Recognition module.
This project implements a simple CNN model to classify images into nine categories using the RealWaste dataset. Additionally, it compares the performance of this custom CNN with pre-trained models (ResNet50 and DenseNet121) to evaluate trade-offs, advantages, and limitations.

## Dataset Overview

The dataset used for this project is RealWaste, consisting of images categorized into the following classes:

- **Cardboard**
- **Food Organics**
- **Glass**
- **Metal**
- **Miscellaneous Trash**
- **Paper**
- **Plastic**
- **Textile Trash**
- **Vegetation**

### Dataset Splits

- **Training Set:** 2851 images (60%)
- **Validation Set:** 950 images (20%)
- **Test Set:** 951 images (20%)

## Custom CNN Model

### Environment Setup

Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Data Preparation

- **Dataset Path:** Ensure the dataset is located at `./realwaste/realwaste-main/RealWaste`.
- **Transformations:**
  - Resize images to `128x128`.
  - Normalize with mean and standard deviation of `(0.5, 0.5, 0.5)`.

### Model Architecture

The custom CNN model consists of the following components:

1. **Two Convolutional Layers:**
   - Filters: 32 and 64
   - Kernel size: 3x3
   - Activation: ReLU
   - MaxPooling after each layer

2. **Fully Connected Layers:**
   - Hidden Units: 128
   - Dropout: 0.5
   - Output Units: 9 (for classification)

### Training Process

- **Optimizer:** Adam with a learning rate of 0.001
- **Loss Function:** CrossEntropyLoss
- **Number of Epochs:** 20
- **Batch Size:** 32

## Pre-trained Models

### ResNet

- **Architecture:** ResNet50 pre-trained on ImageNet
- **Modifications:** Final fully connected layer replaced to match the number of classes (9).

### DenseNet

- **Architecture:** DenseNet121 pre-trained on ImageNet
- **Modifications:** Final classifier layer replaced to match the number of classes (9).

### Fine-tuning

Both models were fine-tuned with the same training, validation, and testing splits. The Adam optimizer and CrossEntropyLoss were used, with training conducted for 20 epochs.

## Results and Comparisons

| Model        | Test Loss | Test Accuracy |
|--------------|-----------|---------------|
| Custom CNN   | 1.4439    | 65.47%        |
| ResNet50     | 0.8071    | 72.74%        |
| DenseNet121  | 0.7478    | 74.42%        |

### Performance Observations

- Pre-trained models achieved higher accuracy and lower loss compared to the custom CNN.
- DenseNet121 outperformed ResNet50 in both test loss and accuracy.

## Discussion and Trade-offs

### Custom Model

- **Advantages:** Lightweight, customizable, and suitable for resource-constrained devices.
- **Limitations:** Requires more data for training and has limited generalization capabilities.

### Pre-trained Models

- **Advantages:** Faster convergence, better generalization, and ability to handle complex patterns.
- **Limitations:** Computationally intensive, larger model size, and sensitivity to hyperparameter tuning.

## Requirements

- **Python:** 3.8+
- **PyTorch:** 1.13+
- **torchvision:** 0.14+
- **Other Libraries:**
  - matplotlib
  - seaborn
  - scikit-learn
