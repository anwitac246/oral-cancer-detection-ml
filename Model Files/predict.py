import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('final_mobilenet_model.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict the class of an image
def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]  # Output is a single value (sigmoid)
    
    # Interpret prediction
    if prediction < 0.5:
        return f"Prediction: Benign (Confidence: {1 - prediction:.2%})"
    else:
        return f"Prediction: Cancer (Confidence: {prediction:.2%})"

# Test the model with an image
image_path = r'archive (1)\Oral Cancer\Oral Cancer Dataset\NON CANCER\005.jpeg'  # Replace with the path to your test image
result = predict_image(image_path)
print(result)
