# Dataset Documentation for Tomato Disease AI Application

## Dataset Structure
The dataset is organized into the following directories:

- `train/`: Contains images for training the model, organized by disease category.
- `test/`: Contains images for testing the model, also organized by disease category.
- `validation/`: Contains images for validating the model's performance during training.

## Image Format
All images in the dataset are in JPEG format and should be named according to the following convention:
```
<disease_name>_<unique_id>.jpg
```
Where `<disease_name>` corresponds to the type of tomato disease (e.g., "healthy", "early_blight", "late_blight", etc.) and `<unique_id>` is a unique identifier for each image.

## Preprocessing Steps
Before using the dataset for training, the following preprocessing steps should be performed:

1. **Resizing**: All images should be resized to a uniform dimension (e.g., 224x224 pixels) to ensure consistency in input size for the model.
2. **Normalization**: Pixel values should be normalized to a range of [0, 1] by dividing by 255.
3. **Augmentation**: Data augmentation techniques such as rotation, flipping, and zooming can be applied to increase the diversity of the training dataset.

## Usage
To load and preprocess the dataset, refer to the utility functions provided in `src/utils.py`. Ensure that the dataset is properly structured as described above before initiating the training process.