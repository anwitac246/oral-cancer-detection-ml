from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('final_mobilenet_model.keras')
@app.route('/')
def home():
    return "Flask is running successfully!"
def preprocess_image(image) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "Invalid file"}), 400

    try:
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:
            result = {"diagnosis": "Benign", "confidence": f"{(1 - prediction) * 100:.2f}%"}
        else:
            result = {"diagnosis": "Cancer", "confidence": f"{prediction * 100:.2f}%"}

        return jsonify(result)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
