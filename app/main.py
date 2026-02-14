from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model_utils import predict_image

app = FastAPI(title="Deepfake Detection API")

@app.get("/")
def root():
    return {"message": "Deepfake API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": f"{confidence:.2%}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)