# PixelProof
PixelProof is a Deepfake Detection website where you can check if a video or photo is real or if it has been tampered with. It’s built for anyone who wants to make sure what they are seeing online is the truth.

## 🐍 Option 1: Run Locally with Python

1. Clone the repository:

```bash
git clone https://github.com/TheinMinHtet/PixelProof
cd PixelProof
```

2 Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3.Install dependencies:
```bash
pip install -r requirements.txt
```

4.Run the FastAPI server:
```bash
python -m app/main.py
```

5.Open the API docs in your browser:
```bash
http://localhost:8000/docs
```
------------------------------------
## 🐳 Option 2: Run with Docker

1.Clone the repository:
```bash
git clone https://github.com/TheinMinHtet/PixelProof
cd PixelProof
```

2.Build the Docker image:
```bash
docker build -t PixelProof .
```

3.Run the Docker container:
```bash
docker run -p 8000:8000 -v "C:/Users/Asus/Desktop/weights:/app/weights" PixelProof
```
4.Access API docs:
```bash
http://localhost:8000/docs
```
## 🧠 Model Weights
The deepfake detection model weights (`best_fusion_srm_model.pth`) are too large for GitHub. You must download them separately from Hugging Face.

1. Visit the [DeepFake_Images_Detection](https://huggingface.co/Thein777/DeepFake_Images_Detection/tree/main).
2. Download the `best_fusion_srm_model.pth` file.
3. Create a folder in the project root: `app/weights/`.
4. Place the downloaded file inside that folder.

## How to call from frontend
1.Do the steps that i provided and then check http://127.0.0.1:8000 it shows {Deepfake API is running}.

2.then use http://127.0.0.1:8000/predict as a endpoint.

## Training Code 
If you want to know how to train this model more details you can check in this (https://huggingface.co/Thein777/DeepFake_Images_Detection/tree/main) in this you can also see the weights of the model

## Training Dataset
If you want to check the training dataset you can check in this (https://huggingface.co/datasets/Thein777/real_fake_low_high_quality_images/tree/main)
