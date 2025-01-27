# TFLiteSegmentation

This repository contains code and resources for performing image segmentation using TensorFlow Lite.

## Getting Started

Follow the instructions below to set up and run the project.

### Prerequisites

- Python 3.x
- TensorFlow
- TensorFlow Lite

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Intelligentedgesystems/Benchmark-Repositories.git
    cd Benchmark-Repositories/TFLiteSegmentation
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Download Model Files

Download the model file "lite-model_deeplabv3_1_metadata_2.tflite" for the above repository from the following link:
[Download Model Files](https://drive.google.com/drive/folders/1z2Kr2W7oyvf-x0Km12uWvLTEZWNKm136?usp=sharing)

### Usage

1. Place the downloaded model files in the `models` directory.
2. Run the segmentation script:
    ```sh
    python segment.py --image_path path/to/your/image.jpg
    ```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- TensorFlow
- TensorFlow Lite
