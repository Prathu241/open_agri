# Open Agri - Plant Disease Detection AI

This project focuses on identifying plant diseases using advanced Deep Learning models. It includes a user-friendly web interface powered by a multimodal Large Language Model (LLM) and specialized pipelines for Tomato Disease detection.

## 📂 Project Structure

- **`new/`**: Contains the interactive web application.
  - `main.py`: A Gradio-based interface utilizing the `YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection` model for analyzing leaf images, describing symptoms, and suggesting treatments.
- **`tomato-disease-ai/`**: A comprehensive pipeline for tomato disease classification and segmentation.
  - `dataset/` & `segmented_dataset/`: Training and validation data directories.
  - `models/`: Storage for trained Keras/H5 models.
  - `src/`: Training scripts (`train.py`), utilities (`utils.py`), and evaluation scripts.
  - `segmentation/` & `patch_classifier/`: Specialized scripts for image segmentation and patch-based classification.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment or Conda environment to manage dependencies.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Prathu241/open_agri.git
    cd open_agri
    ```

2.  **Install General Dependencies (for `tomato-disease-ai`):**
    ```bash
    cd tomato-disease-ai
    pip install -r requirements.txt
    ```

3.  **Install Dependencies for the Web App (`new/`):**
    The web app requires additional libraries like `gradio`, `torch`, and `transformers`.
    ```bash
    pip install gradio torch transformers opencv-python pillow accelerator
    ```

## 🎮 Usage

### 1. Run the Web Application (LLaVA Model)
Start the AI assistant to analyze any plant leaf image.

```bash
cd new
python main.py
```
This will launch a local Gradio server (usually at `http://127.0.0.1:7860`). Upload an image to get a detailed disease analysis.

### 2. Train the Tomato Disease Model
To train the custom classifiers or segmentation models:

```bash
cd tomato-disease-ai/src
python train.py
```
*(Refer to individual scripts in `tomato-disease-ai` for specific segmentation or patch-based training tasks.)*

## 🧠 Models Used

- **LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection**: A fine-tuned multimodal model for comprehensive plant disease diagnosis.
- **Custom CNNs/SegFormer**: Tailored models for high-accuracy tomato leaf disease segmentation and classification.
