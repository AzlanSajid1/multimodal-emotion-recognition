# fusion.py
# Combines audio, video, and NLP predictions into a single emotion dict

def fuse(audio_probs: dict, video_probs: dict, nlp_probs: dict,
         w_video: float = 0.40, w_audio: float = 0.35, w_nlp: float = 0.25) -> dict:

    # normalize all keys to lowercase
    audio_probs = {k.lower(): v for k, v in audio_probs.items()}
    video_probs = {k.lower(): v for k, v in video_probs.items()}
    nlp_probs   = {k.lower(): v for k, v in nlp_probs.items()}

    all_emotions = set(audio_probs) | set(video_probs) | set(nlp_probs)

    # rebalance weights if nlp is absent
    if not nlp_probs:
        total   = w_video + w_audio
        w_video = round(w_video / total, 4)
        w_audio = round(w_audio / total, 4)
        w_nlp   = 0.0

    fused = {}
    for emotion in all_emotions:
        score = (
            w_video * video_probs.get(emotion, 0.0) +
            w_audio * audio_probs.get(emotion, 0.0) +
            w_nlp   * nlp_probs.get(emotion, 0.0)
        )
        fused[emotion] = round(score, 4)

    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))

def top_emotion(fused: dict) -> tuple[str, float]:
    """Returns (emotion_label, confidence) for the top prediction."""
    label = next(iter(fused))
    return label, fused[label]


def format_for_llm(audio_probs: dict, video_probs: dict,
                   nlp_probs: dict, fused: dict) -> str:
    """Builds the structured prompt payload to send to the LLM."""

    def top5(d):
        items = list(d.items())[:5]
        return ", ".join(f"{k} {v:.2f}" for k, v in items)

    top_label, top_conf = top_emotion(fused)

    return f"""A multimodal emotion recognition system analyzed a video/audio file.

Audio signals (top 5):   {top5(audio_probs)}
Video signals (top 5):   {top5(video_probs)}
Text/NLP signals (top 5): {top5(nlp_probs)}
Fused result (top 5):    {top5(fused)}
Dominant emotion: {top_label} (confidence {top_conf:.0%})

Please provide:
1. A brief psychological interpretation of the emotional state (2-3 sentences)
2. Note any significant disagreement between audio, video, and text signals
3. A one-line summary suitable for a report
"""