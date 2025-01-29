# Efficient-Net-Image_Classification_tflite

This repository contains an implementation of EfficientNet for image classification using TensorFlow Lite. The project aims to provide an efficient and lightweight solution for image classification tasks on edge devices.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [License](#license)

## Introduction
EfficientNet is a family of convolutional neural networks that achieve state-of-the-art accuracy while being computationally efficient. This repository provides a TensorFlow Lite implementation of EfficientNet for image classification, making it suitable for deployment on edge devices.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Intelligentedgesystems/Benchmark-Repositories.git
cd Efficient-Net-Image_Classification_tflite
pip install -r requirements.txt
```

## Usage
To use the model for image classification, follow these steps:

1. Prepare your input image.
2. Download the model file "efficientnet-lite0-fp32.tflite" from the repository here: [Google Drive Link](https://drive.google.com/drive/folders/1z2Kr2W7oyvf-x0Km12uWvLTEZWNKm136?usp=sharing)
3. Run the inference.

Example code:

```python
python test.py
```

## Model Details
The EfficientNet model used in this repository is pre-trained on the ImageNet dataset. The TensorFlow Lite model is optimized for performance on edge devices.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
