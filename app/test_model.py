import tensorflow as tf

model_path = "models/lung_model_raw.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("Model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput Details:")
print(input_details)

print("\nOutput Details:")
print(output_details)