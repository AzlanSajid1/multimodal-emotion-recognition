import joblib
import numpy as np
import torch
import torch.nn as nn
import os
# ── MLP architecture (must match checkpoint) ──────────────────────────────────
class EmotionMLP(nn.Module):
    def __init__(self, input_dim=5000, num_classes=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),   # 0
            nn.BatchNorm1d(512),         # 1
            nn.ReLU(),                   # 2
            nn.Dropout(0.3),             # 3
            nn.Linear(512, 256),         # 4
            nn.BatchNorm1d(256),         # 5
            nn.ReLU(),                   # 6
            nn.Dropout(0.3),             # 7
            nn.Linear(256, num_classes)  # 8
        )
    def forward(self, x):
        return self.net(x)


# ── Load all models once at import time ──────────────────────────────────────
#NLP_DIR = r"D:\final_emotion_app\models\nlp"
NLP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nlp")
print("Loading NLP models...")
vectorizer    = joblib.load(f"{NLP_DIR}/tfidf_vectorizer.pkl")
label_encoder = joblib.load(f"{NLP_DIR}/label_encoder.pkl")
xgb_model     = joblib.load(f"{NLP_DIR}/xgb_model.pkl")
svm_model     = joblib.load(f"{NLP_DIR}/svm_model.pkl")

NUM_CLASSES = len(label_encoder.classes_)

mlp_model = EmotionMLP(input_dim=5000, num_classes=NUM_CLASSES)
ck = torch.load(f"{NLP_DIR}/mlp_model.pt", map_location="cpu")
mlp_model.load_state_dict(ck["model"])          # ← unwrap the "model" key
mlp_model.eval()

EMOTION_CLASSES = list(label_encoder.classes_)
print(f"NLP models loaded ✓  ({NUM_CLASSES} classes)")


# ── Main prediction function ──────────────────────────────────────────────────
def predict_text(text: str) -> dict:
    features       = vectorizer.transform([text])
    features_dense = features.toarray().astype(np.float32)

    # XGB — use decision_function path since it may be a pipeline/wrapper
    try:
        xgb_probs = xgb_model.predict_proba(features)[0]
    except AttributeError:
        decision  = xgb_model.decision_function(features)[0]
        decision  = decision - decision.max()
        exp_d     = np.exp(decision)
        xgb_probs = exp_d / exp_d.sum()

    # LinearSVC → softmax over decision_function scores
    decision   = svm_model.decision_function(features)[0]
    decision   = decision - decision.max()
    exp_d      = np.exp(decision)
    svm_probs  = exp_d / exp_d.sum()

    # MLP probabilities
    with torch.no_grad():
        tensor    = torch.tensor(features_dense)
        logits    = mlp_model(tensor)
        mlp_probs = torch.softmax(logits, dim=1).numpy()[0]

    # Average all three
    avg_probs = (xgb_probs + svm_probs + mlp_probs) / 3.0

    result = {
        emotion: round(float(prob), 4)
        for emotion, prob in zip(EMOTION_CLASSES, avg_probs)
    }
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))