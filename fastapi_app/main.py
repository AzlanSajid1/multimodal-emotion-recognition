from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from contextlib import asynccontextmanager
import shutil
import tempfile
import os
import subprocess
import google.generativeai as genai
from pathlib import Path
from functools import lru_cache


import base64
import io
from PIL import Image
import cv2
# ── Lifespan — runs once at startup and shutdown ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────
    print("Loading models...")

    # Import all inference modules — this triggers their module-level
    # model loading code (the code that runs outside any function)
    import audio_inference
    import video_inference
    import nlp_inference

    # Store them on app.state so every endpoint can access them
    app.state.audio_inference = audio_inference
    app.state.video_inference = video_inference
    app.state.nlp_inference   = nlp_inference

    print("All models loaded ✓")

    yield  # ← server runs here, handling requests

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    print("Shutting down...")


# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Emotion Analyzer",
    description="Analyzes emotions from text audio and video",
    version="1.0.0",
    lifespan=lifespan           # ← attach the lifespan
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#static folder serve
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# add this after app.add_middleware(...)
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("fastapi_app/static/index.html")




# ── Pydantic models ───────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotions:    Dict[str, float]
    top_emotion: str
    confidence:  float

class FullAnalysisResponse(BaseModel):
    video:       Dict[str, float]
    audio:       Dict[str, float]
    nlp:         Dict[str, float]
    fused:       Dict[str, float]
    top_emotion: str
    confidence:  float
    llm_analysis: str | None = None   # ← optional, only present if API key given

class GeminiRequest(BaseModel):
    api_key:     str
    video_probs: Dict[str, float]
    audio_probs: Dict[str, float]
    nlp_probs:   Dict[str, float]
    fused_probs: Dict[str, float]
    top_emotion: str
    confidence:  float

class GeminiResponse(BaseModel):
    analysis: str

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(upload: UploadFile, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return tmp.name


def extract_audio(video_path: str) -> str | None:
    audio_path = video_path.replace(os.path.splitext(video_path)[1], ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", "22050", audio_path],
        capture_output=True
    )
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    return None


@lru_cache(maxsize=1)
def get_whisper_model():
    import whisper

    project_root = Path(__file__).resolve().parents[1]
    whisper_dir = project_root / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    return whisper.load_model("base", download_root=str(whisper_dir))


def transcribe_audio(audio_path: str | None) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return ""

    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path, fp16=False)
        return (result.get("text") or "").strip()
    except Exception:
        return ""


def transcribe_audio_with_segments(audio_path: str | None) -> tuple[str, list[dict]]:
    if not audio_path or not os.path.exists(audio_path):
        return "", []

    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path, fp16=False)
        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        return text, segments
    except Exception:
        return "", []


def build_time_windows(duration_sec: float, window_sec: float = 4.0, hop_sec: float = 2.0) -> list[tuple[float, float]]:
    if duration_sec <= 0:
        return []
    if duration_sec <= window_sec:
        return [(0.0, duration_sec)]

    windows = []
    start = 0.0
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        windows.append((start, end))
        if end >= duration_sec:
            break
        start += hop_sec
    return windows


def average_probability_dicts(prob_dicts: list[dict]) -> dict:
    if not prob_dicts:
        return {}
    sums: dict[str, float] = {}
    for probs in prob_dicts:
        for emotion, score in probs.items():
            key = emotion.lower()
            sums[key] = sums.get(key, 0.0) + float(score)
    count = float(len(prob_dicts))
    avg = {k: round(v / count, 4) for k, v in sums.items()}
    return dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))


def extract_media_window(input_path: str, start_sec: float, dur_sec: float, output_path: str) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", input_path,
            "-t", f"{dur_sec:.3f}",
            output_path,
        ],
        capture_output=True
    )
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0


def split_text_into_window_chunks(text: str, num_windows: int) -> list[str]:
    clean_text = text.strip()
    if not clean_text or num_windows <= 0:
        return []
    words = clean_text.split()
    if not words:
        return []
    chunks = []
    chunk_size = max(1, len(words) // num_windows)
    idx = 0
    for i in range(num_windows):
        if i == num_windows - 1:
            piece = words[idx:]
        else:
            piece = words[idx:idx + chunk_size]
        chunks.append(" ".join(piece).strip())
        idx += chunk_size
    return chunks


def transcript_chunk_for_window(segments: list[dict], start_sec: float, end_sec: float) -> str:
    chunk_lines = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if e >= start_sec and s <= end_sec:
            t = (seg.get("text") or "").strip()
            if t:
                chunk_lines.append(t)
    return " ".join(chunk_lines).strip()


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0:
            return float(frames / fps)
        return 0.0
    finally:
        cap.release()


def build_gemini_prompt(video_probs, audio_probs, nlp_probs,
                        fused_probs, top_emotion, confidence,
                        has_transcript=True) -> str:
    def top5(d):
        if not d:
            return "not available"
        items = list(d.items())[:5]
        return ", ".join(f"{k} {v:.2f}" for k, v in items)

    transcript_note = (
        "Text analysis was performed on the provided transcript."
        if has_transcript
        else "No transcript was provided — text modality was excluded from fusion."
    )

    return f"""You are an expert psychologist analyzing multimodal emotion data.

A multimodal emotion recognition system analyzed a video file and produced these results:

Video model signals (top 5):  {top5(video_probs)}
Audio model signals (top 5):  {top5(audio_probs)}
Text/NLP signals (top 5):     {top5(nlp_probs)}
Fused result (top 5):         {top5(fused_probs)}
Dominant emotion: {top_emotion} (confidence {confidence:.0%})

Note: {transcript_note}

Please provide:
1. A brief psychological interpretation of the emotional state (2-3 sentences)
2. Note any significant disagreement between audio and video signals
3. A one-line summary suitable for a report

Keep your response concise and professional."""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Emotion Analyzer API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "audio": "loaded",
            "video": "loaded",
            "nlp":   "loaded"
        }
    }


@app.post("/analyze/text", response_model=EmotionResponse)
def analyze_text(request: TextRequest):
    from fusion import top_emotion
    from nlp_inference import predict_text

    emotions = predict_text(request.text)
    top_label, top_conf = top_emotion(emotions)

    return EmotionResponse(
        emotions=emotions,
        top_emotion=top_label,
        confidence=top_conf
    )


@app.post("/analyze/video", response_model=EmotionResponse)
def analyze_video(file: UploadFile = File(...)):
    from fusion import top_emotion
    from video_inference import predict_video

    allowed = [".mp4", ".avi", ".mov", ".mkv"]
    suffix  = os.path.splitext(file.filename)[1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Allowed: {allowed}"
        )

    video_path = save_upload(file, suffix)
    try:
        emotions = predict_video(video_path, num_frames=16)
        top_label, top_conf = top_emotion(emotions)
        return EmotionResponse(
            emotions=emotions,
            top_emotion=top_label,
            confidence=top_conf
        )
    finally:
        os.remove(video_path)


@app.post("/analyze/audio", response_model=EmotionResponse)
def analyze_audio(file: UploadFile = File(...)):
    from fusion import top_emotion
    from audio_inference import predict_audio

    suffix     = os.path.splitext(file.filename)[1].lower()
    audio_path = save_upload(file, suffix)
    try:
        emotions = predict_audio(audio_path)
        top_label, top_conf = top_emotion(emotions)
        return EmotionResponse(
            emotions=emotions,
            top_emotion=top_label,
            confidence=top_conf
        )
    finally:
        os.remove(audio_path)


@app.post("/analyze/full", response_model=FullAnalysisResponse)
def analyze_full(
    file:       UploadFile = File(...),
    transcript: str        = Form(default=""),
    gemini_key: str        = Form(default=""),
    w_video:    float      = Form(default=0.40),
    w_audio:    float      = Form(default=0.35),
    w_nlp:      float      = Form(default=0.25)
):
    from fusion import fuse, top_emotion
    from video_inference import predict_video
    from audio_inference import predict_audio
    from nlp_inference   import predict_text

    suffix       = os.path.splitext(file.filename)[1].lower()
    video_path   = save_upload(file, suffix)
    audio_path   = None
    windows_dir  = tempfile.mkdtemp(prefix="temporal_windows_")

    try:
        audio_path = extract_audio(video_path)
        duration_sec = get_video_duration(video_path)
        windows = build_time_windows(duration_sec, window_sec=4.0, hop_sec=2.0)
        if not windows:
            windows = [(0.0, 0.0)]

        provided_transcript = transcript.strip()
        auto_transcript = ""
        whisper_segments: list[dict] = []

        if not provided_transcript:
            auto_transcript, whisper_segments = transcribe_audio_with_segments(audio_path)

        transcript_text = provided_transcript or auto_transcript
        has_transcript = bool(transcript_text)

        text_chunks = (
            split_text_into_window_chunks(transcript_text, len(windows))
            if provided_transcript else []
        )

        window_video_probs = []
        window_audio_probs = []
        window_nlp_probs = []
        window_fused_probs = []

        for idx, (start_sec, end_sec) in enumerate(windows):
            dur_sec = max(0.05, end_sec - start_sec)

            # ── Video window inference ───────────────────────────────────────
            v_out = os.path.join(windows_dir, f"video_win_{idx}.mp4")
            video_probs_w = {}
            if extract_media_window(video_path, start_sec, dur_sec, v_out):
                try:
                    video_probs_w = predict_video(v_out, num_frames=8)
                except Exception:
                    video_probs_w = {}

            # ── Audio window inference ───────────────────────────────────────
            a_out = os.path.join(windows_dir, f"audio_win_{idx}.wav")
            audio_probs_w = {}
            if audio_path and extract_media_window(audio_path, start_sec, dur_sec, a_out):
                try:
                    audio_probs_w = predict_audio(a_out)
                except Exception:
                    audio_probs_w = {}

            # ── Text window inference ────────────────────────────────────────
            nlp_probs_w = {}
            if provided_transcript and idx < len(text_chunks):
                chunk = text_chunks[idx]
            else:
                chunk = transcript_chunk_for_window(whisper_segments, start_sec, end_sec)
            if chunk:
                try:
                    nlp_probs_w = predict_text(chunk)
                except Exception:
                    nlp_probs_w = {}

            fused_w = fuse(
                audio_probs_w, video_probs_w, nlp_probs_w,
                w_video=w_video, w_audio=w_audio, w_nlp=w_nlp
            )

            if video_probs_w:
                window_video_probs.append(video_probs_w)
            if audio_probs_w:
                window_audio_probs.append(audio_probs_w)
            if nlp_probs_w:
                window_nlp_probs.append(nlp_probs_w)
            if fused_w:
                window_fused_probs.append(fused_w)

        video_probs = average_probability_dicts(window_video_probs)
        audio_probs = average_probability_dicts(window_audio_probs)
        nlp_probs = average_probability_dicts(window_nlp_probs)
        fused = average_probability_dicts(window_fused_probs)
        if not fused:
            fused = fuse(audio_probs, video_probs, nlp_probs, w_video=w_video, w_audio=w_audio, w_nlp=w_nlp)
        top_label, top_conf = top_emotion(fused)

        # ── Optional Gemini analysis ──────────────────────────────────────────
        llm_analysis = None
        if gemini_key.strip():
            try:
                genai.configure(api_key=gemini_key.strip())
                model  = genai.GenerativeModel("gemini-2.5-flash")
                prompt = build_gemini_prompt(
                    video_probs, audio_probs, nlp_probs,
                    fused, top_label, top_conf, has_transcript= has_transcript
                )
                response     = model.generate_content(prompt)
                llm_analysis = response.text
            except Exception as e:
                err_str = str(e).lower()
                if "api_key" in err_str or "invalid" in err_str or "credential" in err_str:
                    llm_analysis = "❌ Invalid API key. Check your key at aistudio.google.com"
                elif "quota" in err_str or "limit" in err_str or "429" in err_str:
                    llm_analysis = "❌ Quota exceeded. Your free tier limit is reached for today."
                elif "permission" in err_str or "403" in err_str:
                    llm_analysis = "❌ Permission denied. Make sure the Gemini API is enabled in your Google project."
                elif "network" in err_str or "connection" in err_str:
                    llm_analysis = "❌ Network error. Check your internet connection."
                else:
                    llm_analysis = f"❌ Gemini error: {str(e)}"
        else:
            llm_analysis = "⚠️ No API key provided. Enter your Gemini key in the settings panel."   

        return FullAnalysisResponse(
            video=video_probs,
            audio=audio_probs,
            nlp=nlp_probs,
            fused=fused,
            top_emotion=top_label,
            confidence=top_conf,
            llm_analysis=llm_analysis
        )

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        shutil.rmtree(windows_dir, ignore_errors=True)


@app.post("/analyze/frames")
def analyze_frames(file: UploadFile = File(...)):
    """Returns per-frame predictions + base64 encoded frame images."""
    from video_inference import predict_frame, transform
    import torch

    allowed = [".mp4", ".avi", ".mov", ".mkv"]
    suffix  = os.path.splitext(file.filename)[1].lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{suffix}'")

    video_path = save_upload(file, suffix)

    try:
        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step   = max(1, total // 16)
        frames = []

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs = predict_frame(pil)

            # encode frame as base64 JPEG
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=55)
            b64 = base64.b64encode(buf.getvalue()).decode()

            frames.append({"probs": probs, "img": b64})
            if len(frames) >= 16:
                break

        cap.release()
        return {"frames": frames}

    finally:
        os.remove(video_path)

@app.post("/analyze/llm", response_model=GeminiResponse)
def analyze_llm(request: GeminiRequest):
    """Standalone endpoint — pass already-computed probs, get LLM analysis back."""
    try:
        genai.configure(api_key=request.api_key)
        model    = genai.GenerativeModel("gemini-2.5-flash")
        prompt   = build_gemini_prompt(
            request.video_probs, request.audio_probs,
            request.nlp_probs,   request.fused_probs,
            request.top_emotion, request.confidence
        )
        response = model.generate_content(prompt)
        return GeminiResponse(analysis=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")