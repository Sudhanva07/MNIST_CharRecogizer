import os
import pickle
import boto3
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model from S3
s3 = boto3.client("s3",region_name="ap-southeast-2")
s3.download_file(
    "mnist-model-storage-024757002421-ap-southeast-2-an",
    "model.pkl",
    "/tmp/model.pkl"
)
with open("/tmp/model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    pixels = np.array(body['pixels'], dtype=np.float32)

    if pixels.shape != (784,):
        return jsonify(error="Expected 784 pixel values"), 400

    prediction = model.predict(pixels.reshape(1, -1))
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)