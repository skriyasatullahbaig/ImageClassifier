from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model_path = os.path.join(os.getcwd(), "ImageClassifier", "cat_dog_breed_classifier.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# Auto-extract class labels from dataset folder
dataset_path = "dataset/archive/train"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

class_labels = sorted(os.listdir(dataset_path))  # Extracts class names from folders
print("Loaded Class Labels:", class_labels)  # Debugging output


# Prediction function
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)

    # Debugging output
    print("Raw Predictions:", predictions)
    print("Predicted Index:", class_index)

    if class_index >= len(class_labels):  # Prevents index errors
        return "Unknown Breed"

    return class_labels[class_index]


# Home Route
@app.route("/")
def home():
    return render_template("index.html")


# API Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)

    if not os.path.exists("uploads"):
        os.makedirs("uploads")  # Create uploads folder if not exists

    file.save(file_path)

    prediction = predict_breed(file_path)
    return jsonify({"prediction": prediction})


# Run Flask server
if __name__ == "__main__":
    app.run(debug=True)
