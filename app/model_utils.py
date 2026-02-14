import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipForImageClassification
import torchvision.transforms as T
import os

# =====================================================
# 1. SETUP
# =====================================================

DEVICE = torch.device("cpu")  # CPU for deployment
SIGLIP_ID = "prithivMLmods/deepfake-detector-model-v1"
EMB_DIM = 128


# =====================================================
# 2. ARCHITECTURE (MUST MATCH TRAINING EXACTLY)
# =====================================================

class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        filter1 = [[0, 0, 0], [1, -1, 0], [0, 0, 0]]
        filter2 = [[0, 1, 0], [0, -1, 0], [0, 0, 0]]
        filter3 = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
        q = [filter1, filter2, filter3]
        filter_kernel = torch.FloatTensor(q).unsqueeze(1).repeat(1, 3, 1, 1)
        self.weight = nn.Parameter(filter_kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=1, padding=1)


class ArtifactCNN(nn.Module):
    def __init__(self, output_dim=EMB_DIM):
        super().__init__()
        self.srm = SRMLayer()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.srm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SigLIP_ArtifactNet(nn.Module):
    def __init__(self, siglip_model_name=SIGLIP_ID):
        super().__init__()

        self.sigl_model = SiglipForImageClassification.from_pretrained(siglip_model_name)
        self.artifact_cnn = ArtifactCNN(output_dim=EMB_DIM)

        sigl_dim = self.sigl_model.config.vision_config.hidden_size

        self.fusion_classifier = nn.Sequential(
            nn.Linear(sigl_dim + EMB_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values, images_cnn, is_lq):
        outputs = self.sigl_model.base_model.vision_model(pixel_values=pixel_values)
        sigl_emb = outputs.last_hidden_state[:, 0, :]

        cnn_emb = self.artifact_cnn(images_cnn)

        mask = is_lq.unsqueeze(1).float()
        cnn_emb = cnn_emb * mask

        combined = torch.cat([sigl_emb, cnn_emb], dim=1)
        return self.fusion_classifier(combined)


# =====================================================
# 3. GLOBAL MODEL LOADING (LOAD ONCE)
# =====================================================

print("🔄 Loading Deepfake Model...")

processor = AutoImageProcessor.from_pretrained(SIGLIP_ID)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = SigLIP_ArtifactNet().to(DEVICE)

weights_path = os.path.join(
    os.path.dirname(__file__),
    "weights",
    "best_fusion_srm_model.pth"
)

if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Disable gradients for safety
    for param in model.parameters():
        param.requires_grad = False

    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(
        f"CRITICAL: Weights file not found at {weights_path}"
    )


# =====================================================
# 4. INFERENCE FUNCTION
# =====================================================

@torch.no_grad()
def predict_image(pil_image):

    sigl_in = processor(
        images=pil_image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    cnn_in = transform(pil_image).unsqueeze(0).to(DEVICE)

    # Better LQ check (use smallest dimension)
    is_lq = torch.tensor(
        [1.0 if min(pil_image.size) < 600 else 0.0]
    ).to(DEVICE)

    out = model(sigl_in, cnn_in, is_lq)
    probs = F.softmax(out, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    label = "FAKE" if pred.item() == 1 else "REAL"

    return label, confidence.item()