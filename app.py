import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify,render_template

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html', k=5, train_size=60000)

@app.route('/predict', methods=['POST'])
def predict():
    body=request.get_json(force=True)
    pixels = np.array(body['pixels'], dtype=np.float32)
    print(pixels.shape)
    if pixels.shape != (784,):
        return jsonify(error="Expected 784 pixel values"), 400
    
    pixels = pixels.reshape(1, -1)  # Reshape to (1, 784)1

    prediction = model.predict(pixels)
    print(f"Prediction: {prediction}")
    return jsonify({"prediction": int(prediction[0])})
    

 
if __name__ == "__main__":
    app.run(debug=True, port=5000)