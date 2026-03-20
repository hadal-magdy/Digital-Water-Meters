# ============================================================
# smart_app.py  v3
# Run:  python smart_app.py
# Open: http://127.0.0.1:5000
# ============================================================

import re
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

app     = Flask(__name__)
history = []
_ocr    = None

def get_ocr():
    global _ocr
    if _ocr is None:
        import easyocr
        print("⏳  Loading EasyOCR...")
        _ocr = easyocr.Reader(['en'], gpu=False)
        print("✅  EasyOCR ready!")
    return _ocr


# ── Find LCD rectangle ────────────────────────────────────────
def find_lcd_rect(img):
    """
    Find the bright rectangular LCD display inside the meter.
    Returns (x1, y1, x2, y2) of the best candidate, or None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Boost contrast
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Find bright regions (LCD screen is bright)
    _, bright = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)

    # Morphology to fill gaps inside LCD
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed  = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best      = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area  = cw * ch
        ratio = cw / max(ch, 1)

        # LCD displays are: wide rectangles, medium size, in center area
        if (area < 0.005 * w * h or area > 0.5 * w * h):
            continue
        if ratio < 1.5 or ratio > 10:
            continue
        # Should be roughly in the upper-center of the meter
        cx_rel = (x + cw/2) / w
        cy_rel = (y + ch/2) / h
        if cx_rel < 0.2 or cx_rel > 0.9:
            continue

        # Score: prefer wider, brighter, more centered
        brightness = np.mean(enhanced[y:y+ch, x:x+cw])
        score = area * (brightness / 255) * (1 - abs(cx_rel - 0.5))

        if score > best_score:
            best_score = score
            best = (x, y, x+cw, y+ch)

    return best


# ── Read meter from image bytes ───────────────────────────────
def read_meter(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w  = img.shape[:2]
    ocr   = get_ocr()
    annotated = img.copy()

    # ── Step 1: Find LCD rectangle ────────────────────────────
    lcd_rect = find_lcd_rect(img)
    lcd_crop = None

    if lcd_rect:
        x1, y1, x2, y2 = lcd_rect
        # Add padding
        pad = 8
        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(w, x2+pad)
        y2p = min(h, y2+pad)

        lcd_crop = img[y1p:y2p, x1p:x2p]

        # Draw green rectangle on annotated image
        cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
        cv2.putText(annotated, "LCD Display", (x1p, y1p-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ── Step 2: Run OCR on LCD crop first, then full image ────
    all_results = []

    # Priority 1: OCR on LCD crop (upscaled)
    if lcd_crop is not None and lcd_crop.size > 0:
        # Upscale LCD crop for better OCR
        scale    = max(1, 300 // min(lcd_crop.shape[:2]))
        lcd_big  = cv2.resize(lcd_crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        # Also try grayscale + threshold versions
        gray_lcd = cv2.cvtColor(lcd_big, cv2.COLOR_BGR2GRAY)
        _, thresh_lcd = cv2.threshold(gray_lcd, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bgr = cv2.cvtColor(thresh_lcd, cv2.COLOR_GRAY2BGR)

        for variant in [lcd_big, thresh_bgr]:
            results = ocr.readtext(variant)
            for bbox, text, conf in results:
                cleaned = re.sub(r'[^0-9.]', '', text)
                if len(cleaned) >= 2:
                    all_results.append((cleaned, conf, "lcd"))
                    print(f"  [LCD OCR] text={text!r}  cleaned={cleaned!r}  conf={conf:.2f}")

    # Priority 2: OCR on full image (upscaled)
    scale_full = max(1, 800 // max(h, w))
    img_big    = cv2.resize(img, None, fx=scale_full, fy=scale_full,
                            interpolation=cv2.INTER_CUBIC)
    full_results = ocr.readtext(img_big)

    for bbox, text, conf in full_results:
        cleaned = re.sub(r'[^0-9.]', '', text)
        if len(cleaned) >= 2:
            all_results.append((cleaned, conf, "full"))
            print(f"  [FULL OCR] text={text!r}  cleaned={cleaned!r}  conf={conf:.2f}")

        # Draw all detections on annotated image
        if conf > 0.4:
            pts = np.array([[int(p[0]/scale_full), int(p[1]/scale_full)]
                            for p in bbox], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 200, 255), 1)

    # ── Step 3: Pick best reading ──────────────────────────────
    # Score each candidate
    scored = []
    for text, conf, source in all_results:
        score = conf
        length = len(text.replace(".", ""))

        # Prefer 4-8 digit readings (typical meter readings)
        if 4 <= length <= 8:
            score += 0.5
        elif 3 <= length < 4:
            score += 0.2

        # Prefer readings starting with 0 (like 00907)
        if text.startswith("0"):
            score += 0.3

        # Prefer LCD source
        if source == "lcd":
            score += 0.4

        # Prefer no decimal for main reading
        if "." not in text:
            score += 0.1

        scored.append((score, text, conf, source))

    scored.sort(reverse=True)

    # Draw best reading on image
    if scored:
        best_text = scored[0][1]
        best_conf = scored[0][2]
        cv2.putText(annotated, f"READING: {best_text} m3",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
    else:
        best_text = None
        best_conf = 0.0

    # Encode annotated image
    _, buf  = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # All unique candidates
    seen   = set()
    unique = []
    for _, text, conf, source in scored:
        if text not in seen:
            seen.add(text)
            unique.append({
                "text":       text,
                "confidence": round(conf, 2),
                "source":     source
            })

    return {
        "reading":    best_text,
        "confidence": round(best_conf, 2),
        "lcd_found":  lcd_rect is not None,
        "all_reads":  unique[:10],
        "image":      img_b64
    }


# ── Dashboard ─────────────────────────────────────────────────
DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Smart Meter Reader</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0f14;--surface:#111720;--border:#1e2a38;
  --accent:#00c2ff;--accent2:#00ff9d;--warn:#ffb830;
  --danger:#ff4f4f;--text:#c8d8e8;--muted:#4a6070;
  --mono:'Share Tech Mono',monospace;--sans:'Barlow',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-weight:300;min-height:100vh}
header{background:var(--surface);border-bottom:1px solid var(--border);padding:16px 32px;display:flex;align-items:center;justify-content:space-between}
.logo-text{font-size:14px;letter-spacing:3px;text-transform:uppercase;color:var(--accent);font-weight:600}
.logo-sub{font-size:11px;letter-spacing:2px;color:var(--muted);margin-top:2px}
.badge{background:rgba(0,255,157,.1);color:var(--accent2);font-size:11px;padding:4px 12px;border-radius:20px;font-family:var(--mono)}
main{padding:28px 32px;max-width:1200px;display:grid;grid-template-columns:1fr 1fr;gap:20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:22px 24px}
.card-label{font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
.card-full{grid-column:1/3}
.upload-zone{border:2px dashed var(--border);border-radius:10px;padding:36px;text-align:center;cursor:pointer;transition:all .2s;position:relative}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(0,194,255,.04)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-icon{font-size:32px;margin-bottom:8px;opacity:.5}
.upload-text{font-size:13px;color:var(--muted);line-height:1.8}
.upload-text span{color:var(--accent)}
#preview-wrap{display:none;margin-top:14px}
#preview-img{width:100%;border-radius:8px;border:1px solid var(--border)}
.btn{margin-top:14px;width:100%;padding:13px;background:var(--accent);color:#000;font-family:var(--mono);font-size:14px;letter-spacing:2px;border:none;border-radius:8px;cursor:pointer;transition:opacity .2s;display:none}
.btn:hover{opacity:.85}
.btn:disabled{opacity:.4;cursor:not-allowed}
#result-wrap{display:none}
.reading-big{font-family:var(--mono);font-size:60px;color:var(--accent);letter-spacing:8px;line-height:1;margin:8px 0;animation:flash .4s ease}
@keyframes flash{from{color:#fff}to{color:var(--accent)}}
.reading-unit{font-size:12px;color:var(--muted);margin-top:4px}
.lcd-badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-family:var(--mono);margin-top:10px}
.lcd-found{background:rgba(0,255,157,.1);color:var(--accent2)}
.lcd-not{background:rgba(255,184,48,.1);color:var(--warn)}
.read-row{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border-bottom:1px solid var(--border);font-family:var(--mono);font-size:13px}
.read-row:last-child{border-bottom:none}
.read-row.best{background:rgba(0,194,255,.08);border-radius:6px;margin-bottom:2px}
.source-lcd{color:var(--accent2);font-size:10px}
.source-full{color:var(--muted);font-size:10px}
#annotated-img{width:100%;border-radius:8px;border:1px solid var(--border);display:none}
.hist-table{width:100%;border-collapse:collapse;font-size:13px}
.hist-table th{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;font-weight:400}
.hist-table td{padding:9px 10px;border-bottom:1px solid var(--border);font-family:var(--mono)}
.hist-table tr:last-child td{border-bottom:none}
.empty{text-align:center;padding:32px;color:var(--muted);font-size:13px}
.spinner{display:none;margin:16px auto;width:28px;height:28px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-msg{color:var(--danger);font-size:13px;margin-top:10px;display:none}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo-text">💧 AquaVision Smart</div>
    <div class="logo-sub">Universal Water Meter Reader</div>
  </div>
  <span class="badge">EasyOCR + LCD Detection</span>
</header>
<main>
  <div class="card">
    <div class="card-label">Upload Meter Image</div>
    <div class="upload-zone" id="drop-zone">
      <input type="file" id="file-input" accept="image/*">
      <div class="upload-icon">📷</div>
      <div class="upload-text">
        Drop any water meter image<br>
        or <span>click to browse</span>
      </div>
    </div>
    <div id="preview-wrap"><img id="preview-img" src="" alt="preview"></div>
    <div class="spinner" id="spinner"></div>
    <div class="error-msg" id="error-msg"></div>
    <button class="btn" id="btn-detect" onclick="detect()">⚡ READ METER</button>
  </div>

  <div class="card" id="result-wrap">
    <div class="card-label">Meter Reading</div>
    <div class="reading-big" id="reading-big">—</div>
    <div class="reading-unit">m³ · cubic metres</div>
    <div id="lcd-badge"></div>
    <div style="margin-top:18px">
      <div class="card-label">All Detected Numbers</div>
      <div id="all-reads"></div>
    </div>
  </div>

  <div class="card card-full" id="annotated-wrap" style="display:none">
    <div class="card-label">Annotated Image — Green box = LCD region detected</div>
    <img id="annotated-img" src="" alt="annotated">
  </div>

  <div class="card card-full">
    <div class="card-label">Detection History</div>
    <div id="history-area">
      <div class="empty">No detections yet</div>
    </div>
  </div>
</main>

<script>
let selectedFile = null;
let historyData  = [];

const fileInput = document.getElementById('file-input');
const dropZone  = document.getElementById('drop-zone');

fileInput.addEventListener('change', e => { if(e.target.files[0]) handleFile(e.target.files[0]); });
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src            = e.target.result;
    document.getElementById('preview-wrap').style.display = 'block';
    document.getElementById('btn-detect').style.display   = 'block';
    document.getElementById('error-msg').style.display    = 'none';
    document.getElementById('annotated-wrap').style.display = 'none';
    document.getElementById('result-wrap').style.display    = 'none';
  };
  reader.readAsDataURL(file);
}

async function detect() {
  if (!selectedFile) return;
  const btn     = document.getElementById('btn-detect');
  const spinner = document.getElementById('spinner');
  const errMsg  = document.getElementById('error-msg');
  btn.disabled          = true;
  btn.textContent       = 'Reading...';
  spinner.style.display = 'block';
  errMsg.style.display  = 'none';
  try {
    const fd = new FormData();
    fd.append('image', selectedFile);
    const res  = await fetch('/detect', { method:'POST', body:fd });
    const data = await res.json();
    if (data.error) {
      errMsg.textContent   = '❌ ' + data.error;
      errMsg.style.display = 'block';
    } else {
      showResult(data);
    }
  } catch(e) {
    errMsg.textContent   = '❌ Server error.';
    errMsg.style.display = 'block';
  } finally {
    btn.disabled    = false;
    btn.textContent = '⚡ READ METER';
    spinner.style.display = 'none';
  }
}

function showResult(data) {
  document.getElementById('result-wrap').style.display = 'block';
  const bigEl = document.getElementById('reading-big');
  bigEl.textContent = data.reading || '—';
  bigEl.style.animation = 'none'; void bigEl.offsetWidth;
  bigEl.style.animation = 'flash .4s ease';

  document.getElementById('lcd-badge').innerHTML = data.lcd_found
    ? '<span class="lcd-badge lcd-found">✅ LCD region detected</span>'
    : '<span class="lcd-badge lcd-not">⚠️ LCD not found — used full image</span>';

  document.getElementById('all-reads').innerHTML =
    data.all_reads.map((r,i) => `
      <div class="read-row ${i===0?'best':''}">
        <span style="color:${i===0?'var(--accent)':'var(--text)'};font-size:${i===0?'18px':'14px'}">${r.text}</span>
        <span>
          <span class="${r.source==='lcd'?'source-lcd':'source-full'}">${r.source==='lcd'?'LCD':'Full'}</span>
          <span style="color:var(--muted);margin-left:8px">${Math.round(r.confidence*100)}%</span>
          ${i===0?'<span style="color:var(--accent2);margin-left:8px">← best</span>':''}
        </span>
      </div>`).join('');

  if (data.image) {
    document.getElementById('annotated-wrap').style.display = 'block';
    const img = document.getElementById('annotated-img');
    img.src   = 'data:image/jpeg;base64,' + data.image;
    img.style.display = 'block';
  }

  historyData.unshift({
    reading:   data.reading,
    lcd_found: data.lcd_found,
    conf:      Math.round(data.confidence*100),
    time:      new Date().toLocaleTimeString(),
    file:      selectedFile.name
  });
  renderHistory();
}

function renderHistory() {
  const area = document.getElementById('history-area');
  if (!historyData.length) { area.innerHTML='<div class="empty">No detections yet</div>'; return; }
  area.innerHTML = `
    <table class="hist-table">
      <thead><tr><th>#</th><th>Reading</th><th>LCD Found</th><th>Confidence</th><th>File</th><th>Time</th></tr></thead>
      <tbody>${historyData.map((r,i)=>`
        <tr>
          <td>${historyData.length-i}</td>
          <td style="color:var(--accent);font-size:16px">${r.reading}</td>
          <td style="color:${r.lcd_found?'var(--accent2)':'var(--warn)'}">${r.lcd_found?'✅ Yes':'⚠️ No'}</td>
          <td>${r.conf}%</td>
          <td style="color:var(--muted);font-size:11px">${r.file.substring(0,30)}</td>
          <td>${r.time}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

document.addEventListener('keydown', e => { if(e.key==='Enter'&&selectedFile) detect(); });
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return DASHBOARD

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    img_bytes = request.files["image"].read()
    if not img_bytes:
        return jsonify({"error": "Empty file."}), 400
    result = read_meter(img_bytes)
    if result is None:
        return jsonify({"error": "Could not read image."}), 400
    if not result.get("reading"):
        return jsonify({"error": "No numbers detected. Try a clearer image."}), 200
    print(f"📥  reading={result['reading']}  lcd={result['lcd_found']}  conf={result['confidence']}")
    return jsonify(result)


# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  💧  AquaVision Smart v3")
    print("="*50)
    print("  Open →  http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=False, port=5000)