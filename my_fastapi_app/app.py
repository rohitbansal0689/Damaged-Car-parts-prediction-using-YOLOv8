from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import onnxruntime as ort
import io

app = FastAPI()

# Load ONNX model once
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name

# Dummy labels — replace with actual if needed
LABELS = ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess
    image_resized = cv2.resize(image, (640, 640))
    image_transposed = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
    image_normalized = image_transposed / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: image_batch})

    # Return raw output (optionally format properly)
    return JSONResponse({
        "outputs": [o.tolist() for o in outputs]
    })
