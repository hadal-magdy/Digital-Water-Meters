import re
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        import easyocr
        print("⏳ Loading EasyOCR Engine...")
        _ocr = easyocr.Reader(['en'], gpu=False)
        print("✅ Engine Ready")
    return _ocr

# ── Step 1: Region Detection (The LCD Finder) ────────────────
def get_candidate_regions(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # spatial exclusion: ignores the top 35% (serial numbers) 
    # and bottom 15% (technical labels)
    y_start, y_end = int(h * 0.35), int(h * 0.85)
    focus_mask = np.zeros((h, w), dtype=np.uint8)
    focus_mask[y_start:y_end, :] = 255
    gray = cv2.bitwise_and(gray, focus_mask)

    # use blackhat to find dark rectangles on light backgrounds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # otsu thresholding finds the best contrast automatically
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # close gaps to unify the lcd box
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / float(max(rh, 1))
        area_pct = (rw * rh) / (h * w)
        
        # valid lcds are horizontal and take up a reasonable portion of the center
        if 1.8 < aspect < 6.5 and 0.005 < area_pct < 0.25:
            regions.append((x, y, rw, rh))
            
    # sort by proximity to the horizontal center line
    regions.sort(key=lambda r: abs((r[1] + r[3]/2) - (h/2)))
    return regions[:3]

# ── Step 2: Intelligent OCR & Filtering ──────────────────────
def ocr_and_filter(img, regions):
    ocr = get_ocr()
    h_img, w_img = img.shape[:2]
    candidates = []

    for x, y, rw, rh in regions:
        # crop with padding
        pad = 20
        crop = img[max(0, y-pad):min(h_img, y+rh+pad), max(0, x-pad):min(w_img, x+rw+pad)]
        if crop.size == 0: continue
        
        # sharpen the crop to make lcd segments clearer
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray_crop, -1, sharpen_kernel)
        
        # upscale for better accuracy
        upscale = cv2.resize(sharpened, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # allow digits and dots
        results = ocr.readtext(upscale, allowlist='0123456789. ')
        
        for (_, text, prob) in results:
            clean = text.replace(" ", "").strip()
            if len(clean) < 3: continue
            
            score = prob
            # center-y bias: higher scores for items closer to the horizontal center line
            y_center_norm = (y + rh/2) / h_img
            dist_from_mid = abs(y_center_norm - 0.6) 
            score += (1.0 - dist_from_mid)
            
            # bonus for meter-like patterns
            if "." in clean: score += 0.5
            if clean.startswith("00"): score += 0.3
            
            candidates.append({"text": clean, "score": score, "box": (x,y,rw,rh)})

    if not candidates: return None, None, []
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]
    return best['text'], best['box'], candidates

# ── Dashboard UI ──────────────────────────────────────────────
DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AquaVision Smart v8</title>
    <style>
        :root { --bg:#0b0f14; --surface:#111720; --accent:#00c2ff; --border:#1e2a38; --text:#c8d8e8; }
        body { background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; }
        header { background:var(--surface); padding:20px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
        main { display:grid; grid-template-columns: 1fr 1fr; gap:20px; padding:20px; max-width:1200px; margin:auto; }
        .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }
        .full { grid-column: 1/3; }
        .upload-box { border:2px dashed var(--border); padding:40px; text-align:center; border-radius:10px; cursor:pointer; }
        #preview, #annotated { width:100%; border-radius:8px; margin-top:15px; border:1px solid var(--border); }
        .reading { font-size:54px; color:var(--accent); letter-spacing:5px; text-align:center; margin:20px 0; font-family:monospace; }
        .btn { width:100%; padding:15px; background:var(--accent); border:none; border-radius:8px; cursor:pointer; font-weight:600; }
    </style>
</head>
<body>
    <header>
        <div><div style="color:var(--accent); letter-spacing:2px; font-weight:bold;">AQUAVISION SMART V8</div><div style="font-size:11px; color:#4a6070;">Spatial Masking + Sharpness Filter</div></div>
    </header>
    <main>
        <div class="card">
            <div class="upload-box" onclick="document.getElementById('inp').click()">
                <input type="file" id="inp" hidden onchange="loadFile(event)">
                <div style="font-size:30px">📸</div>
                <div>Drop or Click to Upload</div>
            </div>
            <img id="preview" style="display:none">
            <button class="btn" id="run-btn" style="display:none; margin-top:10px;" onclick="process()">ANALYZE METER</button>
        </div>
        <div class="card">
            <div style="font-size:10px; letter-spacing:2px; color:#4a6070;">DETECTED READING</div>
            <div class="reading" id="res-val">----</div>
        </div>
        <div class="card full" id="out-card" style="display:none">
            <div style="font-size:10px; letter-spacing:2px; color:#4a6070;">DETECTION VERIFICATION (GREEN BOX = LCD)</div>
            <img id="annotated">
        </div>
    </main>
    <script>
        let currentFile = null;
        function loadFile(e) {
            currentFile = e.target.files[0];
            document.getElementById('preview').src = URL.createObjectURL(currentFile);
            document.getElementById('preview').style.display = 'block';
            document.getElementById('run-btn').style.display = 'block';
        }
        async function process() {
            const btn = document.getElementById('run-btn');
            btn.innerText = 'PROCESSING...';
            const fd = new FormData();
            fd.append('image', currentFile);
            const res = await fetch('/detect', {method:'POST', body:fd});
            const data = await res.json();
            document.getElementById('res-val').innerText = data.reading || 'FAILED';
            document.getElementById('annotated').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('out-card').style.display = 'block';
            btn.innerText = 'ANALYZE METER';
        }
    </script>
</body>
</html>"""

# ── Flask Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return DASHBOARD

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file: return jsonify({"error": "No image"}), 400
    
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    regions = get_candidate_regions(img)
    reading, box, all_reads = ocr_and_filter(img, regions)
    
    # annotate image for verification
    annotated = img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 212, 255), 3)

    _, buf = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "reading": reading,
        "image": img_b64
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)