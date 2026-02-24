import os
import io
import cv2
import torch
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torch.nn.functional import softmax

# ==========================
# IMPORT MODELS
# ==========================
from app.model_utils import predict_image
from app.model3d import VideoResNet2D
from fastapi.middleware.cors import CORSMiddleware

# ==========================
# CREATE ONE FASTAPI APP
# ==========================
app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# VIDEO MODEL SETTINGS
# ==========================
CLIP_LEN = 8
SIZE = 224
STRIDE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best_model.pt")

video_model = VideoResNet2D(num_classes=2).to(DEVICE)

if os.path.exists(MODEL_PATH):
    video_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    video_model.eval()
    print(f"✅ Loaded video weights from {MODEL_PATH}")
else:
    print("❌ Video model weights not found!")

# ==========================
# ROOT
# ==========================
@app.get("/")
def root():
    return {"message": "Deepfake API is running"}

# ==========================
# IMAGE PREDICTION
# ==========================
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": f"{confidence:.2%}"
    }

# ==========================
# VIDEO PROCESSING
# ==========================
def extract_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return torch.zeros(1, 3, CLIP_LEN, SIZE, SIZE, device=DEVICE)

    indices = list(range(0, total_frames, max(1, STRIDE)))

    if len(indices) >= CLIP_LEN:
        start = (len(indices) - CLIP_LEN) // 2
        indices = indices[start:start + CLIP_LEN]
    else:
        indices = indices[:CLIP_LEN]

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return torch.zeros(1, 3, CLIP_LEN, SIZE, SIZE, device=DEVICE)

    if len(frames) < CLIP_LEN:
        frames += [frames[-1]] * (CLIP_LEN - len(frames))

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

    clip_t = []
    for img in frames:
        img = cv2.resize(img, (SIZE, SIZE))
        t = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
        t = (t - mean) / std
        clip_t.append(t)

    clip_tensor = torch.stack(clip_t, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return clip_tensor.contiguous()

# ==========================
# VIDEO PREDICTION
# ==========================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        shutil.copyfileobj(file.file, temp)
        temp.close()

        input_tensor = extract_clip(temp.name)

        with torch.no_grad():
            output = video_model(input_tensor)
            probs = softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        label = "FAKE" if pred.item() == 1 else "REAL"

        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence.item() * 100:.2f}%"
        }

    finally:
        os.unlink(temp.name)

# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
