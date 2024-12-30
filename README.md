# Oral Cancer Detection Project

This project aims to provide early detection of oral cancer using a machine learning model built with TensorFlow's MobileNetV2 architecture and trained on a custom dataset of oral cancer images. Users can upload images via a responsive web interface built using Next.js and Tailwind CSS, with Flask and Python managing the backend for image processing and prediction.

---

## Key Features

- **Responsive Web Interface**: Built using **Next.js** and styled with **Tailwind CSS**, ensuring seamless usage across devices.
- **Backend in Flask**: Flask serves as the backend for handling image uploads and managing communication with the ML model.
- **MobileNetV2-based Model**: A fine-tuned version of MobileNetV2 from ImageNet, designed for efficient image classification tasks.
- **Custom Dataset**: The model was trained on a dataset of oral cancer images with augmentation to ensure robust predictions.
- **High Accuracy**: The model achieves over **95% accuracy** in detecting oral cancer from uploaded images.
- **Real-time Predictions**: Users can upload images and receive instant diagnoses (Benign or Cancer) along with confidence scores.

---

## Machine Learning Model Details

The ML model for detecting oral cancer was implemented as follows:

1. **Architecture**:
    - Used **MobileNetV2** pretrained on ImageNet as the base model.
    - Added custom layers:
        - Global Average Pooling.
        - Fully Connected Dense Layer (128 neurons with ReLU activation).
        - Dropout Layer (to prevent overfitting).
        - Output Layer with a sigmoid activation for binary classification.

2. **Training**:
    - Dataset divided into 80% training and 20% validation using `image_dataset_from_directory`.
    - Data augmentation applied to improve generalization.
    - Binary cross-entropy loss function used.
    - Optimized with Adam optimizer (learning rate: 0.0001).

3. **Class Balancing**:
    - Class weights computed to handle any imbalance in the dataset between benign and cancerous cases.

4. **Callbacks**:
    - Early stopping to halt training if validation loss does not improve.
    - Model checkpointing to save the best model during training.
    - Learning rate reduction on plateau.

5. **Performance**:
    - Achieved **95% accuracy** with low validation loss on the test set.
    - Model visualized with accuracy and loss plots over training epochs.

6. **Saved Model**:
    - The best-performing model was saved as `final_mobilenet_model.keras`.

---

## Technologies Used

### Frontend
- **Next.js**: For creating a dynamic and efficient user interface.
- **Tailwind CSS**: For responsive and visually appealing designs.

### Backend
- **Flask**: For handling image upload and serving the prediction API.
- **TensorFlow**: For training and deploying the ML model.

### Dataset
The dataset was organized into labeled directories for "Cancer" and "Benign" cases. Images were augmented to create a robust training set.

---

## Usage

### 1. Run the Backend
- Ensure the `final_mobilenet_model.keras` file is present in the backend directory.
- Start the Flask server:
  ```bash
  python prediction.py
