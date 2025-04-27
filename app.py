from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
from PIL import Image
import numpy as np
from tensorflow import keras

app = FastAPI()

# Load model TensorFlow
model = keras.models.load_model('my_model.h5')

# Hàm tiền xử lý ảnh
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Resize đúng kích thước input model
    img_array = np.array(image) / 255.0  # Normalize về [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# API nhận ảnh và dự đoán
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    tensor = transform_image(img_bytes)
    prediction = model.predict(tensor)
    predicted_class = int(np.argmax(prediction))

    return {"prediction": predicted_class}

# Run server local nếu cần
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)