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
python app/main.py
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
docker run -p 8000:8000 PixelProof
```
4.Access API docs:
```bash
http://localhost:8000/docs
```
## 🧠 Model Weights
The deepfake detection model weights (`best_fusion_srm_model.pth`) are too large for GitHub. You must download them separately from Hugging Face.

### Option 1: Manual Download (Recommended)
1. Visit the [PixelProof Model on Hugging Face](https://huggingface.co/spaces/Thein777/PixelProof/tree/main).
2. Download the `best_fusion_srm_model.pth` file.
3. Create a folder in the project root: `app/weights/`.
4. Place the downloaded file inside that folder.

### Option 2: Clone from Hugging Face
If you want the entire project including weights directly from Hugging Face:
```bash
# Make sure you have git-lfs installed
git lfs install
git clone [https://huggingface.co/YourUsername/PixelProof](https://huggingface.co/YourUsername/PixelProof)
