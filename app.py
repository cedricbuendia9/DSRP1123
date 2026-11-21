from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

FEATURES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6"
]

@app.route("/")
def home():
    return jsonify({
        "status": "API activa"
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    values = [[data[col] for col in FEATURES]]

    prediction = model.predict(values)

    return jsonify({
        "prediction": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)