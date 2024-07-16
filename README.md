# Lung Cancer Detection System using U-Net

## Overview

This repository contains the code for a Lung Cancer Detection System utilizing the U-Net architecture. The system is designed for accurate classification of lung cancer CT scans from medical images, providing precise results and aiding in early diagnosis.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)


## Background

Lung cancer detection in medical imaging is a critical task that requires precise classification segmentation of tumors. U-Net, with its powerful image segmentation capabilities, is well-suited for this purpose. This project leverages the strengths of U-Net to build an effective lung cancer detection system.

## Features

- **Biomedical Image Segmentation:** Specially designed for accurate segmentation of lung cancer regions.
- **High Accuracy and Precision:** Captures context and precise localization of tumors.
- **Works with Limited Data:** Effective even with smaller datasets, making it suitable for medical imaging scenarios.
- **Robust to Overfitting:** Incorporates data augmentation and dropout strategies.

## Requirements

- Python 3.6+
- TensorFlow 2.0+
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iheb-sassi/Cancer-detection-CNN-Unet.git
   cd Cancer-detection-CNN-Unet
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:** Ensure your dataset is organized with images and corresponding masks in appropriate directories.
2. **Run the training script:**
   ```bash
   python train.py
   ```
3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```

## Dataset

The dataset should include medical images and their corresponding segmentation masks. You can use publicly available lung cancer datasets or your own labeled data.

## Training

To train the model, execute the `train.py` script. Adjust the hyperparameters and training configurations as needed within the script.

## Evaluation

To evaluate the trained model, run the `evaluate.py` script. This will provide metrics such as accuracy, precision, recall, and IoU (Intersection over Union) for the segmentation performance.

## Results

After training and evaluation, the results will be saved in the `results` directory. This includes the model's performance metrics and sample segmented images.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any improvements or features you'd like to see.

