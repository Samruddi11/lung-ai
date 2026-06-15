from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load TFLite model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "lung_model_raw.tflite"
)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        values = [
            float(request.form["subtlety"]),
            float(request.form["internal_structure"]),
            float(request.form["calcification"]),
            float(request.form["sphericity"]),
            float(request.form["margin"]),
            float(request.form["lobulation"]),
            float(request.form["spiculation"]),
            float(request.form["texture"])
        ]

        input_data = np.array([values], dtype=np.float32)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

        score = float(prediction[0][0])

        result = "Malignant" if score >= 0.5 else "Benign"

    return render_template(
        "index.html",
        result=result
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001
    )