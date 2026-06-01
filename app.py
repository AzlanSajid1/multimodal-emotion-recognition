import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import subprocess
import anthropic
import cv2
import json
import base64
import io
from PIL import Image

from audio_inference import predict_audio
from video_inference import predict_video, predict_frame
from nlp_inference import predict_text
from fusion import fuse, top_emotion, format_for_llm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Order Analyzer",
    page_icon="🤪",
    layout="wide"
)

st.title("Multimodal Emotion Analyzer")
st.caption("Upload a video file — audio, facial expressions, and transcript are analyzed together.")


# ── Frame viewer HTML builder ─────────────────────────────────────────────────
def build_frame_viewer_html(frame_data, audio_probs, fused_probs):
    frames_json = json.dumps(frame_data)
    audio_json  = json.dumps(audio_probs)
    fused_json  = json.dumps(fused_probs)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: transparent; color: #1a1a1a; }}
  .wrap {{ padding: 8px 0; }}

  .player {{ background: #f5f5f3; border-radius: 12px;
             border: 0.5px solid rgba(0,0,0,0.1); overflow: hidden; margin-bottom: 12px; }}
  .screen {{ width: 100%; aspect-ratio: 16/9; background: #0a0a0f;
             position: relative; overflow: hidden; }}
  .screen img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .overlay {{ position: absolute; inset: 0; pointer-events: none; }}

  .emotion-badge {{ position: absolute; top: 12px; left: 12px;
                    background: rgba(10,10,15,0.75);
                    border: 0.5px solid rgba(255,255,255,0.15);
                    border-radius: 8px; padding: 8px 12px; }}
  .badge-label {{ font-size: 10px; color: rgba(255,255,255,0.5); margin-bottom: 2px; }}
  .badge-emotion {{ font-size: 22px; font-weight: 500; color: #fff; letter-spacing: -0.3px; }}
  .badge-conf {{ font-size: 11px; color: rgba(255,255,255,0.5); margin-top: 2px; }}

  .frame-counter {{ position: absolute; top: 12px; right: 12px;
                    background: rgba(10,10,15,0.65);
                    border: 0.5px solid rgba(255,255,255,0.12);
                    border-radius: 6px; padding: 5px 10px;
                    font-size: 11px; color: rgba(255,255,255,0.6); font-variant-numeric: tabular-nums; }}

  .minibars {{ position: absolute; bottom: 12px; left: 12px; right: 12px;
               display: flex; align-items: flex-end; gap: 4px; height: 52px; }}
  .minibar-wrap {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; }}
  .minibar {{ width: 100%; border-radius: 2px 2px 0 0; min-height: 2px; transition: height 0.3s ease; }}
  .minibar-lbl {{ font-size: 8px; color: rgba(255,255,255,0.4);
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                  max-width: 100%; text-align: center; }}

  .playbar {{ display: flex; align-items: center; gap: 10px; padding: 10px 14px 6px; }}
  .playbtn {{ width: 30px; height: 30px; border-radius: 50%;
              background: #1a1a1a; color: #fff; border: none;
              cursor: pointer; font-size: 11px; display: flex;
              align-items: center; justify-content: center; flex-shrink: 0; }}
  .playbtn:hover {{ background: #333; }}
  .frame-lbl {{ font-size: 12px; color: #666; }}

  .timeline {{ padding: 4px 14px 14px; }}
  .tl-hint {{ font-size: 10px; color: #999; margin-bottom: 6px; }}
  .scrubber {{ width: 100%; -webkit-appearance: none; appearance: none;
               height: 4px; border-radius: 2px; background: #ddd; outline: none; cursor: pointer; }}
  .scrubber::-webkit-slider-thumb {{ -webkit-appearance: none; width: 15px; height: 15px;
                                      border-radius: 50%; background: #1a1a1a;
                                      border: 2px solid #fff; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
  .tick-row {{ display: flex; justify-content: space-between; margin-top: 4px; }}
  .tick {{ font-size: 9px; color: #bbb; }}

  .panels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px; }}
  .panel {{ background: #fff; border: 0.5px solid rgba(0,0,0,0.1);
            border-radius: 12px; padding: 14px 16px; }}
  .panel-title {{ font-size: 11px; color: #999; margin-bottom: 10px; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
  .bar-name {{ font-size: 11px; color: #333; width: 62px; flex-shrink: 0; text-transform: capitalize; }}
  .bar-track {{ flex: 1; background: #f0f0ee; border-radius: 2px; height: 7px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.35s ease; }}
  .bar-pct {{ font-size: 11px; color: #999; width: 34px; text-align: right; flex-shrink: 0; font-variant-numeric: tabular-nums; }}

  .chart-card {{ background: #fff; border: 0.5px solid rgba(0,0,0,0.1);
                 border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
  .chart-title {{ font-size: 11px; color: #999; margin-bottom: 10px; }}
  .mod-tabs {{ display: flex; gap: 6px; margin-bottom: 10px; }}
  .mod-tab {{ font-size: 11px; padding: 4px 10px; border-radius: 20px;
              border: 0.5px solid #ddd; background: transparent;
              color: #666; cursor: pointer; }}
  .mod-tab.active {{ background: #1a1a1a; color: #fff; border-color: #1a1a1a; }}
  .chart-wrap {{ position: relative; height: 180px; }}
  .cursor-line {{ position: absolute; top: 0; bottom: 0; width: 1.5px;
                  background: #1a1a1a; pointer-events: none; transition: left 0.2s ease; opacity: 0.4; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
  .leg-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; color: #666; text-transform: capitalize; }}
  .leg-dot {{ width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }}
</style>
</head>
<body>
<div class="wrap">

  <div class="player">
    <div class="screen">
      <img id="frameImg" src="" alt="video frame">
      <div class="overlay">
        <div class="emotion-badge">
          <div class="badge-label">dominant emotion</div>
          <div class="badge-emotion" id="badgeEmotion">—</div>
          <div class="badge-conf" id="badgeConf">—</div>
        </div>
        <div class="frame-counter" id="frameCounter">1 / 16</div>
        <div class="minibars" id="minibars"></div>
      </div>
    </div>

    <div class="playbar">
      <button class="playbtn" id="playBtn">&#9654;</button>
      <span class="frame-lbl" id="frameLbl">Frame 1 of 16</span>
    </div>

    <div class="timeline">
      <div class="tl-hint">drag to scrub · click chart to jump</div>
      <input type="range" class="scrubber" id="scrubber" min="0" max="15" value="0" step="1">
      <div class="tick-row" id="tickRow"></div>
    </div>
  </div>

  <div class="panels">
    <div class="panel">
      <div class="panel-title">video model · frame <span id="vFrameN">1</span></div>
      <div id="videoBars"></div>
    </div>
    <div class="panel">
      <div class="panel-title">audio model · overall</div>
      <div id="audioBars"></div>
    </div>
    <div class="panel">
      <div class="panel-title">fused · frame <span id="fFrameN">1</span></div>
      <div id="fusedBars"></div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">emotion confidence across all frames</div>
    <div class="mod-tabs" id="modTabs">
      <button class="mod-tab active" data-mod="video">video</button>
      <button class="mod-tab" data-mod="fused">fused</button>
    </div>
    <div class="chart-wrap">
      <canvas id="tlChart" role="img" aria-label="Line chart of emotion probabilities across 16 frames"></canvas>
      <div class="cursor-line" id="cursorLine"></div>
    </div>
    <div class="legend" id="chartLegend"></div>
  </div>

</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const frameData  = {frames_json};
const audioProbs = {audio_json};
const fusedData  = {fused_json};

const COLORS = {{
  fear:     '#E24B4A',
  disgust:  '#639922',
  surprise: '#EF9F27',
  sad:      '#378ADD',
  neutral:  '#888780',
  happy:    '#1D9E75',
  anger:    '#D4537E',
  angry:    '#D4537E',
  calm:     '#5DCAA5',
  annoyance:'#BA7517',
  excitement:'#AFA9EC',
}};

function getColor(e) {{ return COLORS[e.toLowerCase()] || '#aaa'; }}

function sortedEntries(obj) {{
  return Object.entries(obj).sort((a,b)=>b[1]-a[1]);
}}

let currentFrame = 0;
let playing = false;
let playTimer = null;
let currentMod = 'video';
let tlChart;

// ── frame image ───────────────────────────────────────────────────────────────
function showFrame(fi) {{
  const fd = frameData[fi];
  document.getElementById('frameImg').src = 'data:image/jpeg;base64,' + fd.img;
  const top = sortedEntries(fd.probs)[0];
  document.getElementById('badgeEmotion').textContent = top[0];
  document.getElementById('badgeEmotion').style.color = getColor(top[0]);
  document.getElementById('badgeConf').textContent = Math.round(top[1]*1000)/10 + '% confidence';
  document.getElementById('frameCounter').textContent = (fi+1) + ' / ' + frameData.length;
  document.getElementById('frameLbl').textContent = 'Frame ' + (fi+1) + ' of ' + frameData.length;
  document.getElementById('vFrameN').textContent = fi+1;
  document.getElementById('fFrameN').textContent = fi+1;
  updateMinibars(fd.probs);
  makeBars('videoBars', fd.probs);
  makeBars('audioBars', audioProbs);
  makeBars('fusedBars', fd.fused || fusedData);
  updateCursor(fi);
}}

// ── minibars overlay ──────────────────────────────────────────────────────────
function updateMinibars(probs) {{
  const mb = document.getElementById('minibars');
  mb.innerHTML = '';
  sortedEntries(probs).forEach(([e,v]) => {{
    const wrap = document.createElement('div'); wrap.className='minibar-wrap';
    const bar  = document.createElement('div'); bar.className='minibar';
    bar.style.height = Math.round(v*48)+'px';
    bar.style.background = getColor(e);
    const lbl  = document.createElement('div'); lbl.className='minibar-lbl';
    lbl.textContent = e.slice(0,4);
    wrap.appendChild(bar); wrap.appendChild(lbl);
    mb.appendChild(wrap);
  }});
}}

// ── sidebar bars ─────────────────────────────────────────────────────────────
function makeBars(id, probs) {{
  const el = document.getElementById(id); el.innerHTML='';
  sortedEntries(probs).slice(0,6).forEach(([e,v]) => {{
    const row  = document.createElement('div'); row.className='bar-row';
    const name = document.createElement('div'); name.className='bar-name'; name.textContent=e;
    const track= document.createElement('div'); track.className='bar-track';
    const fill = document.createElement('div'); fill.className='bar-fill';
    fill.style.width = Math.round(v*100)+'%';
    fill.style.background = getColor(e);
    const pct  = document.createElement('div'); pct.className='bar-pct';
    pct.textContent = Math.round(v*100)+'%';
    track.appendChild(fill);
    row.appendChild(name); row.appendChild(track); row.appendChild(pct);
    el.appendChild(row);
  }});
}}

// ── timeline chart ────────────────────────────────────────────────────────────
function buildChart(mod) {{
  const frames = mod==='video'
    ? frameData.map(f=>f.probs)
    : frameData.map(f=>f.fused || fusedData);

  const allEmotions = [...new Set(frames.flatMap(f=>Object.keys(f)))];
  const topEmotions = allEmotions
    .sort((a,b) => {{
      let sa = frames.reduce((s,f)=>s+(f[a]||0),0);
      let sb = frames.reduce((s,f)=>s+(f[b]||0),0);
      return sb-sa;
    }}).slice(0,6);

  const datasets = topEmotions.map(e => ({{
    label: e,
    data: frames.map(f => Math.round((f[e]||0)*1000)/10),
    borderColor: getColor(e),
    backgroundColor: getColor(e)+'22',
    borderWidth: 1.5,
    pointRadius: 3,
    pointHoverRadius: 5,
    fill: false,
    tension: 0.35,
  }}));

  if(tlChart) tlChart.destroy();
  tlChart = new Chart(document.getElementById('tlChart'), {{
    type: 'line',
    data: {{ labels: frameData.map((_,i)=>i+1), datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      animation: {{ duration: 250 }},
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: c=>c.dataset.label+': '+c.parsed.y.toFixed(1)+'%' }} }} }},
      scales: {{
        x: {{ ticks: {{ font: {{size:10}} }}, grid: {{ color:'rgba(0,0,0,0.05)' }} }},
        y: {{ min:0, max:100, ticks: {{ callback:v=>v+'%', font:{{size:10}} }},
              grid: {{ color:'rgba(0,0,0,0.05)' }} }},
      }},
      onClick: (_,els) => {{ if(els.length) goTo(els[0].index); }}
    }}
  }});

  const leg = document.getElementById('chartLegend'); leg.innerHTML='';
  topEmotions.forEach(e => {{
    const item=document.createElement('div'); item.className='leg-item';
    const dot=document.createElement('div'); dot.className='leg-dot';
    dot.style.background=getColor(e);
    item.appendChild(dot); item.appendChild(document.createTextNode(e));
    leg.appendChild(item);
  }});

  setTimeout(()=>updateCursor(currentFrame), 120);
}}

function updateCursor(fi) {{
  if(!tlChart || !tlChart.chartArea) return;
  const ca = tlChart.chartArea;
  const w  = ca.right - ca.left;
  const x  = ca.left + (fi / (frameData.length-1)) * w;
  const pct= (x / document.getElementById('tlChart').offsetWidth)*100;
  document.getElementById('cursorLine').style.left = pct.toFixed(1)+'%';
}}

// ── navigation ────────────────────────────────────────────────────────────────
function goTo(fi) {{
  currentFrame = Math.max(0, Math.min(fi, frameData.length-1));
  document.getElementById('scrubber').value = currentFrame;
  showFrame(currentFrame);
}}

document.getElementById('scrubber').addEventListener('input', e => goTo(parseInt(e.target.value)));

document.getElementById('playBtn').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('playBtn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
  if(playing) {{
    playTimer = setInterval(() => {{
      const next = currentFrame + 1;
      if(next >= frameData.length) {{ playing=false; document.getElementById('playBtn').innerHTML='&#9654;'; clearInterval(playTimer); return; }}
      goTo(next);
    }}, 650);
  }} else {{
    clearInterval(playTimer);
  }}
}});

document.querySelectorAll('.mod-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.mod-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    currentMod = tab.dataset.mod;
    buildChart(currentMod);
  }});
}});

// ── tick marks ───────────────────────────────────────────────────────────────
const tickRow = document.getElementById('tickRow');
frameData.forEach((_,i) => {{
  const t = document.createElement('span'); t.className='tick';
  t.textContent = i+1; tickRow.appendChild(t);
}});

// ── init ──────────────────────────────────────────────────────────────────────
showFrame(0);
buildChart('video');
</script>
</body>
</html>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="Get yours at console.anthropic.com")
    st.markdown("---")
    st.markdown("**Fusion weights**")
    w_video = st.slider("Video weight", 0.0, 1.0, 0.40, 0.05)
    w_audio = st.slider("Audio weight", 0.0, 1.0, 0.35, 0.05)
    w_nlp   = st.slider("Text weight",  0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    transcript_input = st.text_area("Paste transcript (optional)",
                                    placeholder="If you have a transcript, paste it here. "
                                                "Otherwise leave blank.")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded and st.button("▶ Analyze", type="primary"):

    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    audio_path = tmp_path.replace(suffix, ".wav")

    try:
        # ── Step 1: Extract audio ─────────────────────────────────────────────
        with st.spinner("Extracting audio..."):
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path,
                 "-ac", "1", "-ar", "22050", audio_path],
                capture_output=True
            )
            audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0

        # ── Step 2: Run all three models ──────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.spinner("Analyzing video frames..."):
                video_probs = predict_video(tmp_path, num_frames=16)
            st.subheader("🎥 Video")
            for emotion, score in list(video_probs.items())[:5]:
                st.progress(score, text=f"{emotion}: {score:.1%}")

        with col2:
            with st.spinner("Analyzing audio..."):
                if audio_ok:
                    audio_probs = predict_audio(audio_path)
                else:
                    st.warning("Audio extraction failed — skipping audio model.")
                    audio_probs = {}
            st.subheader("🎙️ Audio")
            for emotion, score in list(audio_probs.items())[:5]:
                st.progress(score, text=f"{emotion}: {score:.1%}")

        with col3:
            with st.spinner("Analyzing text..."):
                transcript = transcript_input.strip()
                if not transcript:
                    transcript = "no transcript provided"
                nlp_probs = predict_text(transcript)
            st.subheader("📝 Text")
            for emotion, score in list(nlp_probs.items())[:5]:
                st.progress(score, text=f"{emotion}: {score:.1%}")

        # ── Step 3: Fusion ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔀 Fused Result")

        fused = fuse(audio_probs, video_probs, nlp_probs)
        top_label, top_conf = top_emotion(fused)

        st.metric("Dominant Emotion", top_label.upper(), f"{top_conf:.1%} confidence")

        cols = st.columns(min(5, len(fused)))
        for i, (emotion, score) in enumerate(list(fused.items())[:5]):
            with cols[i]:
                st.metric(emotion, f"{score:.1%}")

        # ── Step 4: Frame-by-frame interactive viewer ─────────────────────────
        st.markdown("---")
        st.subheader("🎞️ Frame-by-Frame Analysis")

        with st.spinner("Building frame viewer..."):
            from video_inference import transform
            import torch

            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step  = max(1, total // 16)
            frame_data = []

            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # per-frame video prediction
                frame_probs = predict_frame(pil)

                # per-frame fused (video + overall audio + overall nlp)
                frame_fused = fuse(audio_probs, frame_probs, nlp_probs)

                # encode frame as base64 JPEG
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=55)
                b64 = base64.b64encode(buf.getvalue()).decode()

                frame_data.append({
                    "probs": frame_probs,
                    "fused": frame_fused,
                    "img":   b64,
                })
                if len(frame_data) >= 16:
                    break
            cap.release()

        components.html(
            build_frame_viewer_html(frame_data, audio_probs, fused),
            height=1100,
            scrolling=True
        )

        # ── Step 5: LLM analysis ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🤖 AI Analysis")

        if not api_key:
            st.info("Enter your Anthropic API key in the sidebar to get an AI narrative analysis.")
        else:
            with st.spinner("Generating analysis..."):
                prompt  = format_for_llm(audio_probs, video_probs, nlp_probs, fused)
                client  = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
                )
                analysis = message.content[0].text
            st.markdown(analysis)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)