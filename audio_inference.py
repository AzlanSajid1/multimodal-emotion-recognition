import joblib
import numpy as np
import torch
import torch.nn as nn
import os
# ── Audio CNN (exact architecture from checkpoint) ────────────────────────────
# Keys: conv1, bn1, conv2, bn2, pool(AdaptiveAvgPool→70), fc1, bn3, fc2
class OptimizedAudioCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(70)      # 128 * 70 = 8960
        self.fc1   = nn.Linear(128 * 70, 256)
        self.bn3   = nn.BatchNorm1d(256)            # bn3 is after fc1
        self.fc2   = nn.Linear(256, num_classes)
        self.drop  = nn.Dropout(0.3)
        self.act   = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)                 # (batch, 8960)
        x = self.drop(self.act(self.bn3(self.fc1(x))))
        return self.fc2(x)


# ── Load all models once at import time ──────────────────────────────────────
#AUDIO_DIR = r"D:\final_emotion_app\models\audio"
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "audio")
print("Loading audio models...")
voting_model   = joblib.load(f"{AUDIO_DIR}/best_voting_ensemble.joblib")
stacking_model = joblib.load(f"{AUDIO_DIR}/best_stacking_ensemble.joblib")
label_encoder  = joblib.load(f"{AUDIO_DIR}/label_encoder.joblib")

cnn_model = OptimizedAudioCNN(num_classes=len(label_encoder.classes_))
cnn_model.load_state_dict(torch.load(
    f"{AUDIO_DIR}/audio_model_cnn_weights.pth",
    map_location="cpu"
))
cnn_model.eval()
print("Audio models loaded ✓")

EMOTION_CLASSES = list(label_encoder.classes_)


# ── Feature extraction from audio file ───────────────────────────────────────
def extract_features(audio_path: str) -> np.ndarray:
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    mel      = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),        # 40
        np.mean(chroma, axis=1),      # 12
        np.mean(mel, axis=1)[:60],    # 60
        np.mean(contrast, axis=1),    #  7
        np.mean(tonnetz, axis=1),     #  6
        np.std(mfcc, axis=1)[:15],    # 15
    ])                                # total = 140

    return features.reshape(1, -1)


# ── Main prediction function ──────────────────────────────────────────────────
def predict_audio(audio_path: str) -> dict:
    features = extract_features(audio_path)

    voting_probs   = voting_model.predict_proba(features)[0]
    stacking_probs = stacking_model.predict_proba(features)[0]

    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32)
        logits = cnn_model(tensor)
        cnn_probs = torch.softmax(logits, dim=1).numpy()[0]

    avg_probs = (voting_probs + stacking_probs + cnn_probs) / 3.0

    result = {
        emotion: round(float(prob), 4)
        for emotion, prob in zip(EMOTION_CLASSES, avg_probs)
    }

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))