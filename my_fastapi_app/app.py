from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']
CONFIDENCE_THRESHOLD = 0.5

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image_resized = cv2.resize(image, (640, 640))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], image_expanded)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = interpreter.get_tensor(output_details[3]['index'])[0]

    results = []
    for i in range(int(num_detections)):
        score = float(scores[i])
        class_id = int(classes[i])
        if score >= CONFIDENCE_THRESHOLD and 0 <= class_id < len(LABELS):
            bbox = boxes[i]
            results.append({
                "label": LABELS[class_id],
                "confidence": round(score, 3),
                "bbox": [round(float(x), 4) for x in bbox]
            })

    return JSONResponse(content={"predictions": results})
