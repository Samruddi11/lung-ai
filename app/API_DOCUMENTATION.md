\# Lung AI Prediction API



\## Base URL



http://127.0.0.1:5001



\---



\## Health Check



GET /



Response:



{

&#x20; "status": "success",

&#x20; "message": "Lung AI Prediction API is running"

}



\---



\## Prediction Endpoint



POST /predict



Request:



{

&#x20; "features": \[1,2,3,4,5,6,7,8]

}



Response:



{

&#x20; "status": "success",

&#x20; "prediction\_score": 0.8075,

&#x20; "prediction\_class": 1,

&#x20; "prediction\_result": "Malignant"

}



\---



\## Model Information



Model Type: TensorFlow Lite (.tflite)



Input Shape: (1,8)



Output Shape: (1,1)



Classes:

\- 0 = Benign

\- 1 = Malignant

