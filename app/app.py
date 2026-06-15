from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model_path = "models/lung_model_raw.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Lung AI Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"status": "error", "message": "features missing"}), 400

    features = data["features"]

    if len(features) != 8:
        return jsonify({"status": "error", "message": "Model expects 8 input features"}), 400

    input_data = np.array([features], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    prediction_class = int(prediction >= 0.5)
    result = "Malignant" if prediction_class == 1 else "Benign"

    return jsonify({
        "status": "success",
        "prediction_score": float(prediction),
        "prediction_class": prediction_class,
        "prediction_result": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)