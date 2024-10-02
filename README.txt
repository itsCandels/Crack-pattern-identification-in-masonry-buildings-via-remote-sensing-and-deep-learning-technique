
# U-Net Crack Detection Project

## Overview

This project focuses on detecting cracks in images using a U-Net model. It consists of three main scripts, each handling a crucial part of the process: training the model, making predictions, and evaluating the model's performance.

### Files:
1. **2_UNET.py** - Script for building, training, and saving the U-Net model.
2. **3_predict.py** - Script for using the trained U-Net model to make predictions on new data.
3. **4_metrics_evaluation.py** - Script for evaluating the performance of the model on test data and calculating evaluation metrics.

---

## Requirements

Before running the scripts, make sure to install the required libraries:

```bash
pip install -r requirements.txt
```

### Required Libraries:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pickle
- Seaborn

---

## 1. U-Net Model Training (`2_UNET.py`)

### Purpose:
This script builds and trains a U-Net model for crack detection. The model is trained on a dataset of crack images and their corresponding masks.

### Usage:

1. **Input Parameters**:
   - Input images: Folder containing the original crack images.
   - Masks: Folder containing the binary masks corresponding to the crack images.
   
2. **Execution**:
   Run the script as follows:
   ```bash
   python 2_UNET.py
   ```

3. **Model Training**:
   - The U-Net model is created using multiple convolutional and max-pooling layers.
   - The script uses callbacks such as `ReduceLROnPlateau` and `EarlyStopping` to optimize the training process.

4. **Model Output**:
   - The trained model is saved as `segmentation_model.h5` in the specified output folder.

---

## 2. Model Prediction (`3_predict.py`)

### Purpose:
This script loads the trained U-Net model and uses it to make predictions on new images.

### Usage:

1. **Input Parameters**:
   - Model file: The saved U-Net model (`segmentation_model.h5`).
   - Input folder: Folder containing the test images on which predictions will be made.
   
2. **Execution**:
   ```bash
   python 3_predict.py
   ```

3. **Predictions**:
   - The model outputs the predicted masks for the input images.
   - The predicted masks are saved in the specified output folder.
   
4. **Visualization**:
   - The script can visualize the original image alongside the predicted mask for validation purposes.

---

## 3. Model Evaluation (`4_metrics_evaluation.py`)

### Purpose:
This script evaluates the performance of the trained model by comparing the predicted masks with the ground truth masks. It calculates various evaluation metrics like IoU, Dice Coefficient, Precision, Recall, and Accuracy.

### Usage:

1. **Input Parameters**:
   - Model file: The saved U-Net model (`segmentation_model.h5`).
   - Images folder: Folder containing the original test images.
   - Masks folder: Folder containing the ground truth masks for evaluation.
   
2. **Execution**:
   ```bash
   python 4_metrics_evaluation.py
   ```

3. **Metrics Computed**:
   - Intersection over Union (IoU)
   - Dice Coefficient (F1 Score)
   - Precision
   - Recall
   - Accuracy

4. **Results**:
   - A detailed report of the evaluation metrics is saved in `evaluation_metrics.txt`.
   - A plot showing the performance of the model across different metrics is generated and saved as `metrics_plot.png`.
   - An aggregated confusion matrix is also generated and saved as `confusion_matrix.png`.

---

## Notes

- Ensure the dataset is properly organized with separate folders for images and masks.
- The training script allows for parameter adjustment such as learning rate, batch size, etc.
- For the best results, experiment with the model's architecture and training parameters.
