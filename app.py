from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
from PIL import Image
import numpy as np
import os
import requests
from tensorflow import keras

app = FastAPI()

model = None  # Ban đầu model chưa load

# Hàm tải model từ Google Drive
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1Z-ePa_UiDWa1KblMLuP6Z0i31C0rceUS"
    r = requests.get(url)
    with open('my_model.h5', 'wb') as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# Hàm tiền xử lý ảnh
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API nhận ảnh và dự đoán
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        if not os.path.exists('my_model.h5'):
            print("Downloading model...")
            download_model()
        model = keras.models.load_model('my_model.h5')
    
    img_bytes = await file.read()
    tensor = transform_image(img_bytes)
    prediction = model.predict(tensor)
    predicted_class = int(np.argmax(prediction))

    return {"prediction": predicted_class}

# Run server local nếu cần
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
