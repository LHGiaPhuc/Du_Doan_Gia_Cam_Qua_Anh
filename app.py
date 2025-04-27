from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
from PIL import Image
import numpy as np
import os
import requests
from tensorflow import keras

app = FastAPI()

# Tự động tải model nếu chưa tồn tại
model_path = 'my_model.h5'
if not os.path.exists(model_path):
    print("Model not found. Downloading from Google Drive...")
    url = "https://drive.google.com/uc?export=download&id=1Z-ePa_UiDWa1KblMLuP6Z0i31C0rceUS"
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# Load model
model = keras.models.load_model(model_path)

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
    img_bytes = await file.read()
    tensor = transform_image(img_bytes)
    prediction = model.predict(tensor)
    predicted_class = int(np.argmax(prediction))

    return {"prediction": predicted_class}

# Chạy server local nếu cần
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
