import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# ── EmotionCNN architecture (must match checkpoint) ───────────────────────────
class ConvBNRELU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()          # ← was "self.skip"
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x)
        )
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)
        self.relu      = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # ── Stage 1: 2× ResidualBlock  64→64 ─────────────────────────────────
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # ── Stage 2: 2× ResidualBlock  64→128 ────────────────────────────────
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )

        # ── Stage 3: ResidualBlock + DepthwiseSep + ResidualBlock  128→256 ───
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            ResidualBlock(256, 256)
        )

        # ── Stage 4: 2× ResidualBlock  256→512 ───────────────────────────────
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )

        # ── Squeeze-and-Excitation  512→32→512 ───────────────────────────────
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # index 0 — not saved (no weights)
            nn.Flatten(),              # index 1 — not saved (no weights)
            nn.Linear(512, 32),        # index 2
            nn.ReLU(inplace=True),     # index 3 — not saved (no weights)
            nn.Linear(32, 512),        # index 4
            nn.Sigmoid()               # index 5 — not saved (no weights)
        )

        # ── Global pool ───────────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Classifier: Linear(512,256) → BN → ReLU → Dropout → Linear ──────
        # indices:          1                2      3      4          5
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 0
            nn.Linear(512, 256),             # 1
            nn.BatchNorm1d(256),             # 2
            nn.ReLU(inplace=True),           # 3
            nn.Dropout(dropout_rate),        # 4
            nn.Linear(256, num_classes)      # 5
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # SE attention — recalibrate channels
        w = self.se(x)                       # (B, 512)
        x = x * w.view(w.size(0), -1, 1, 1) # broadcast over H×W

        x = self.pool(x)                     # (B, 512, 1, 1)
        return self.classifier(x)


# ── Load model ────────────────────────────────────────────────────────────────
VIDEO_DIR = r"D:\final_emotion_app\models\video"

print("Loading video model...")
checkpoint = torch.load(f"{VIDEO_DIR}/emotion_cnn_best.pth", map_location="cpu")

EMOTION_CLASSES = list(checkpoint["idx2label"].values())
NUM_CLASSES     = len(EMOTION_CLASSES)
IDX2LABEL       = checkpoint["idx2label"]

model = EmotionCNN(num_classes=NUM_CLASSES, dropout_rate=0.5)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Video model loaded ✓  ({NUM_CLASSES} classes: {EMOTION_CLASSES})")


# ── Image transform (same as training) ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


# ── Predict on a single PIL image ────────────────────────────────────────────
def predict_frame(pil_image: Image.Image) -> dict:
    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
    result = {IDX2LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ── Predict from a video file (samples frames evenly) ────────────────────────
def predict_video(video_path: str, num_frames: int = 16) -> dict:
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, total // num_frames)

    all_probs = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).numpy()[0]
        all_probs.append(probs)
        if len(all_probs) >= num_frames:
            break

    cap.release()

    avg = np.mean(all_probs, axis=0)
    result = {IDX2LABEL[i]: round(float(avg[i]), 4) for i in range(NUM_CLASSES)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))