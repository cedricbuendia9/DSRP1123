from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({
        "estado": "API activa"
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    X = pd.DataFrame([data])

    prediccion = model.predict(X)

    return jsonify({
        "prediction": float(prediccion[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)