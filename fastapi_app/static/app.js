// ── Emotion colors ────────────────────────────────────────────────────────────
const COLORS = {
  fear:        '#E24B4A',
  disgust:     '#639922',
  surprise:    '#EF9F27',
  sad:         '#378ADD',
  sadness:     '#378ADD',
  neutral:     '#888780',
  happy:       '#1D9E75',
  joy:         '#1D9E75',
  anger:       '#D4537E',
  angry:       '#D4537E',
  calm:        '#5DCAA5',
  annoyance:   '#BA7517',
  excitement:  '#AFA9EC',
  disapproval: '#C0694A',
};
function getColor(e) {
  return COLORS[e.toLowerCase()] || '#7c6af7';
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropzone     = document.getElementById('dropzone');
const fileInput    = document.getElementById('fileInput');
const fileInfo     = document.getElementById('fileInfo');
const fileName     = document.getElementById('fileName');
const fileSize     = document.getElementById('fileSize');
const transcript   = document.getElementById('transcript');
const geminiKey    = document.getElementById('geminiKey');
const analyzeBtn   = document.getElementById('analyzeBtn');
const progressWrap = document.getElementById('progressWrap');
const progressBar  = document.getElementById('progressBar');
const progressLbl  = document.getElementById('progressLabel');
const resultsPanel = document.getElementById('resultsPanel');
const heroName     = document.getElementById('heroName');
const heroConf     = document.getElementById('heroConf');
const errorBox     = document.getElementById('errorBox');
const llmSection   = document.getElementById('llmSection');
const llmText      = document.getElementById('llmText');
const wVideo       = document.getElementById('wVideo');
const wAudio       = document.getElementById('wAudio');
const wNlp         = document.getElementById('wNlp');
const wVideoVal    = document.getElementById('wVideoVal');
const wAudioVal    = document.getElementById('wAudioVal');
const wNlpVal      = document.getElementById('wNlpVal');


let selectedFile = null;

// ── Frame viewer state ────────────────────────────────────────────────────────
let frameData    = [];
let currentFrame = 0;
let playing      = false;
let playTimer    = null;

// ── Slider labels ─────────────────────────────────────────────────────────────
wVideo.addEventListener('input', () => wVideoVal.textContent = parseFloat(wVideo.value).toFixed(2));
wAudio.addEventListener('input', () => wAudioVal.textContent = parseFloat(wAudio.value).toFixed(2));
wNlp.addEventListener('input',   () => wNlpVal.textContent   = parseFloat(wNlp.value).toFixed(2));

// ── Dropzone ──────────────────────────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover',  e  => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  fileName.textContent = f.name;
  fileSize.textContent = (f.size / 1024 / 1024).toFixed(1) + ' MB';
  fileInfo.style.display = 'flex';
  analyzeBtn.disabled = false;
}


// ── Bar rendering ─────────────────────────────────────────────────────────────
function makeBars(containerId, probs, limit = 6) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  Object.entries(probs).slice(0, limit).forEach(([emotion, score]) => {
    const row   = document.createElement('div'); row.className = 'bar-row';
    const name  = document.createElement('div'); name.className = 'bar-name'; name.textContent = emotion;
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill  = document.createElement('div'); fill.className = 'bar-fill';
    fill.style.width      = (score * 100).toFixed(1) + '%';
    fill.style.background = getColor(emotion);
    const pct   = document.createElement('div'); pct.className = 'bar-pct';
    pct.textContent = (score * 100).toFixed(1) + '%';
    track.appendChild(fill);
    row.appendChild(name); row.appendChild(track); row.appendChild(pct);
    el.appendChild(row);
  });
}

function makeFusedChips(probs) {
  const el = document.getElementById('fusedChips');
  el.innerHTML = '';
  Object.entries(probs).slice(0, 8).forEach(([emotion, score], i) => {
    const chip = document.createElement('div');
    chip.className = i === 0 ? 'chip top' : 'chip';
    chip.textContent = `${emotion} ${(score * 100).toFixed(0)}%`;
    if (i === 0) chip.style.background = getColor(emotion);
    el.appendChild(chip);
  });
}

function setProgress(pct, label) {
  progressBar.style.width = pct + '%';
  progressLbl.textContent = label;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = 'block';
  resultsPanel.style.display = 'flex';
}

function hideError() { errorBox.style.display = 'none'; }

// ── Frame viewer ──────────────────────────────────────────────────────────────
function goToFrame(fi) {
  currentFrame = Math.max(0, Math.min(fi, frameData.length - 1));

  const fd  = frameData[currentFrame];
  const top = Object.entries(fd.probs)[0];

  // update image
  document.getElementById('fvImage').src = 'data:image/jpeg;base64,' + fd.img;

  // update badge
  document.getElementById('fvEmotion').textContent = top[0];
  document.getElementById('fvEmotion').style.color = getColor(top[0]);
  document.getElementById('fvConf').textContent    = (top[1] * 100).toFixed(1) + '% confidence';
  document.getElementById('fvCounter').textContent = (currentFrame + 1) + ' / ' + frameData.length;

  // update scrubber
  document.getElementById('fvScrubber').value = currentFrame;

  // update minibars
  const mb = document.getElementById('fvMinibars');
  mb.innerHTML = '';
  Object.entries(fd.probs).forEach(([e, v]) => {
    const wrap = document.createElement('div'); wrap.className = 'fv-minibar-wrap';
    const bar  = document.createElement('div'); bar.className = 'fv-minibar';
    bar.style.height     = Math.round(v * 52) + 'px';
    bar.style.background = getColor(e);
    const lbl  = document.createElement('div'); lbl.className = 'fv-minibar-lbl';
    lbl.textContent = e.slice(0, 4);
    wrap.appendChild(bar); wrap.appendChild(lbl);
    mb.appendChild(wrap);
  });

  // update frame bars panel
  makeBars('fvBars', fd.probs, 7);

  // update frame count label
  document.getElementById('fvFrameLbl').textContent =
    'Frame ' + (currentFrame + 1) + ' of ' + frameData.length;

  // update chart cursor
  updateChartCursor(currentFrame);
}

// ── Timeline chart ────────────────────────────────────────────────────────────
let tlChart = null;

function buildTimelineChart() {
  const ctx = document.getElementById('fvChart').getContext('2d');

  // pick top 5 emotions by average score across all frames
  const emotionTotals = {};
  frameData.forEach(f => {
    Object.entries(f.probs).forEach(([e, v]) => {
      emotionTotals[e] = (emotionTotals[e] || 0) + v;
    });
  });
  const topEmotions = Object.entries(emotionTotals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([e]) => e);

  const datasets = topEmotions.map(e => ({
    label:           e,
    data:            frameData.map(f => ((f.probs[e] || 0) * 100).toFixed(1)),
    borderColor:     getColor(e),
    backgroundColor: getColor(e) + '22',
    borderWidth:     1.5,
    pointRadius:     3,
    pointHoverRadius:5,
    fill:            false,
    tension:         0.35,
  }));

  if (tlChart) tlChart.destroy();
  tlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels:   frameData.map((_, i) => i + 1),
      datasets,
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 250 },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => c.dataset.label + ': ' + c.parsed.y + '%' } },
      },
      scales: {
        x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { min: 0, max: 100, ticks: { callback: v => v + '%', color: '#888', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
      },
      onClick: (_, els) => { if (els.length) goToFrame(els[0].index); },
    },
  });

  // build legend
  const leg = document.getElementById('fvLegend');
  leg.innerHTML = '';
  topEmotions.forEach(e => {
    const item = document.createElement('div'); item.className = 'fv-leg-item';
    const dot  = document.createElement('div'); dot.className  = 'fv-leg-dot';
    dot.style.background = getColor(e);
    item.appendChild(dot);
    item.appendChild(document.createTextNode(e));
    leg.appendChild(item);
  });
}

function updateChartCursor(fi) {
  if (!tlChart || !tlChart.chartArea) return;
  const ca  = tlChart.chartArea;
  const w   = ca.right - ca.left;
  const x   = ca.left + (fi / Math.max(frameData.length - 1, 1)) * w;
  const pct = (x / document.getElementById('fvChart').offsetWidth) * 100;
  document.getElementById('fvCursor').style.left = pct.toFixed(1) + '%';
}

function buildFrameViewer() {
  const section = document.getElementById('frameViewerSection');
  section.style.display = 'block';

  // build tick marks
  const tickRow = document.getElementById('fvTickRow');
  tickRow.innerHTML = '';
  frameData.forEach((_, i) => {
    const t = document.createElement('span');
    t.className   = 'fv-tick';
    t.textContent = i + 1;
    tickRow.appendChild(t);
  });

  // scrubber
  const scrubber = document.getElementById('fvScrubber');
  scrubber.max   = frameData.length - 1;
  scrubber.addEventListener('input', e => goToFrame(parseInt(e.target.value)));

  // play button
  document.getElementById('fvPlayBtn').addEventListener('click', () => {
    playing = !playing;
    document.getElementById('fvPlayBtn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
    if (playing) {
      playTimer = setInterval(() => {
        const next = currentFrame + 1;
        if (next >= frameData.length) {
          playing = false;
          document.getElementById('fvPlayBtn').innerHTML = '&#9654;';
          clearInterval(playTimer);
          return;
        }
        goToFrame(next);
      }, 650);
    } else {
      clearInterval(playTimer);
    }
  });

  buildTimelineChart();
  goToFrame(0);
}
// ── Format Gemini response into clean HTML ────────────────────────────────────
function formatLLMResponse(text) {
  // remove "Here is the analysis:" prefix if present
  text = text.replace(/^here is the analysis[:\s]*/i, '').trim();

  const sections = [
    { num: '1',  label: '🧠 Psychological Interpretation' },
    { num: '2',  label: '⚡ Modality Disagreement'        },
    { num: '3',  label: '📋 Report Summary'               },
  ];

  let html = '';
  let remaining = text;

  sections.forEach((sec, idx) => {
    const nextNum = sections[idx + 1] ? sections[idx + 1].num : null;

    // match "1." or "1:" or "**1.**" style numbering
    const startRe = new RegExp(
      `\\*{0,2}${sec.num}[.):\\s]+\\*{0,2}[^\\n]*\\n?`, 'i'
    );
    const endRe = nextNum
      ? new RegExp(`\\*{0,2}${nextNum}[.):\\s]+`, 'i')
      : null;

    const startMatch = remaining.search(startRe);
    if (startMatch === -1) return;

    let content = remaining.slice(startMatch).replace(startRe, '').trim();
    if (endRe) {
      const endMatch = content.search(endRe);
      if (endMatch !== -1) content = content.slice(0, endMatch).trim();
    }

    // strip leftover bold markers
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // last section gets the accent block style
    if (idx === sections.length - 1) {
      html += `<div class="llm-summary">${content}</div>`;
    } else {
      html += `<h3>${sec.label}</h3><p>${content}</p>`;
    }
  });

  // fallback — if parsing found nothing just show raw text
  if (!html) {
    html = `<p>${text.replace(/\n/g, '<br>')}</p>`;
  }

  return html;
}
// ── Main analyze ──────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  hideError();
  llmSection.style.display          = 'none';
  resultsPanel.style.display        = 'none';
  document.getElementById('frameViewerSection').style.display = 'none';
  progressWrap.style.display        = 'flex';
  analyzeBtn.disabled               = true;
  playing                           = false;
  if (playTimer) clearInterval(playTimer);

  try {
    // ── Call 1: full analysis ─────────────────────────────────────────────────
    setProgress(10, 'Uploading video...');
    const fd1 = new FormData();
    fd1.append('file',       selectedFile);
    fd1.append('transcript', transcript.value.trim());
    fd1.append('gemini_key', geminiKey.value.trim());
    fd1.append('w_video',    wVideo.value);
    fd1.append('w_audio',    wAudio.value);
    fd1.append('w_nlp',      wNlp.value);

    setProgress(25, 'Running all models...');
    const r1 = await fetch('http://localhost:8000/analyze/full', {
      method: 'POST', body: fd1,
    });
    if (!r1.ok) { const e = await r1.json(); throw new Error(e.detail || 'Server error'); }
    const data = await r1.json();

    setProgress(65, 'Building frame viewer...');

    // ── Call 2: frame analysis ────────────────────────────────────────────────
    const fd2 = new FormData();
    fd2.append('file', selectedFile);
    const r2 = await fetch('http://localhost:8000/analyze/frames', {
      method: 'POST', body: fd2,
    });
    if (!r2.ok) { const e = await r2.json(); throw new Error(e.detail || 'Frame error'); }
    const frameResult = await r2.json();
    frameData = frameResult.frames;

    setProgress(95, 'Rendering...');

    setTimeout(() => {
      progressWrap.style.display = 'none';
      resultsPanel.style.display = 'flex';

      // hero
      heroName.textContent      = data.top_emotion;
      heroName.style.color      = getColor(data.top_emotion);
      heroConf.textContent      = (data.confidence * 100).toFixed(1) + '% confidence';

      // modality bars
      makeBars('videoBars', data.video);
      makeBars('audioBars', data.audio);
      makeBars('nlpBars',   data.nlp);

      // fused
      makeFusedChips(data.fused);
      makeBars('fusedBars', data.fused, 8);

      // llm
      llmSection.style.display = 'block';
      if (data.llm_analysis) {
        const isError = data.llm_analysis.startsWith('❌') ||
                        data.llm_analysis.startsWith('⚠️');
        if (isError) {
          llmText.innerHTML   = `<span style="color:#e24b4a">${data.llm_analysis}</span>`;
        } else {
          llmText.innerHTML = formatLLMResponse(data.llm_analysis);
        }
      } else {
        llmText.innerHTML = '<span style="color:#e24b4a">⚠️ No analysis returned.</span>';
      }
      // frame viewer
      buildFrameViewer();

    }, 300);

  } catch (err) {
    progressWrap.style.display = 'none';
    showError('Error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
  }
});