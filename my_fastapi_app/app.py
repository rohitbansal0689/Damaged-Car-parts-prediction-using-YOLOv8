# my_fastapi_app/app.py
import os, io, uuid, random, base64, pathlib
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

APP_VERSION = os.getenv("APP_VERSION", "1.5.0")
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.getenv("MODEL_PATH", str(SCRIPT_DIR / "best.pt"))
MODEL_VERSION = os.getenv("MODEL_VERSION", f"{pathlib.Path(MODEL_PATH).name}@v1")

# thresholds / quality gates (env-tunable)
CONF_DENT = float(os.getenv("CONF_DENT", "0.45"))
CONF_SCRATCH = float(os.getenv("CONF_SCRATCH", "0.50"))
CONF_GENERIC = float(os.getenv("CONF_GENERIC", "0.40"))
SEV_MED = float(os.getenv("SEV_MED", "0.07"))
SEV_HIGH = float(os.getenv("SEV_HIGH", "0.20"))
MIN_SHORT_SIDE = int(os.getenv("MIN_SHORT_SIDE", "640"))
MIN_BLUR_VAR = float(os.getenv("MIN_BLUR_VAR", "60"))
BRIGHT_MIN = float(os.getenv("BRIGHT_MIN", "60"))
BRIGHT_MAX = float(os.getenv("BRIGHT_MAX", "200"))
OVERLAY_DEFAULT = os.getenv("OVERLAY_DEFAULT", "0") == "1"

LABELS = [
    'damaged door', 'damaged window', 'damaged headlight', 'damaged mirror',
    'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield', 'scratch'
]

def set_deterministic(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(False)
            torch.set_num_threads(1)
        except Exception:
            pass

set_deterministic()

# load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model from '{MODEL_PATH}': {e}")

ENABLE_DOCS = os.getenv("ENABLE_DOCS", "0") == "1"

app = FastAPI(
    title="Vehicle Damage Assessment API",
    description="Detect and classify car damage from images.",
    version=os.getenv("APP_VERSION", "1.5.0"),
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
)

def class_conf(label: str) -> float:
    if label == "dent": return CONF_DENT
    if label == "scratch": return CONF_SCRATCH
    return CONF_GENERIC

def assess_quality(img_bgr: np.ndarray) -> List[str]:
    warnings = []
    h, w = img_bgr.shape[:2]
    if min(h, w) < MIN_SHORT_SIDE:
        warnings.append(f"Image is small ({w}x{h}); use ≥{MIN_SHORT_SIDE}px short side.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < BRIGHT_MIN: warnings.append("Image too dark; shoot in better light.")
    if gray.mean() > BRIGHT_MAX: warnings.append("Image too bright; reduce glare/exposure.")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR_VAR:
        warnings.append("Image looks blurry; hold steady or move closer.")
    return warnings

def side_label_from_box(
    img_w: int, img_h: int, x1: int, y1: int, x2: int, y2: int, label: Optional[str] = None
) -> str:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Door-specific wording (frame-relative split at 50%)
    if label and "door" in label.lower():
        return "front door area" if cx < (img_w * 0.5) else "rear door area"

    # Generic fallback (frame-based thirds)
    horiz = "left side" if cx < (img_w / 3.0) else "right side" if cx > (2.0 * img_w / 3.0) else "center"
    vert  = "front area" if cy < (img_h / 3.0) else "rear area" if cy > (2.0 * img_h / 3.0) else "center"
    return vert if vert != "center" else (horiz if horiz != "center" else "side area")




def severity_from_area_ratio(r: float, conf: float) -> str:
    if r >= SEV_HIGH or (r >= SEV_HIGH*0.6 and conf >= 0.75): return "high"
    if r >= SEV_MED: return "medium"
    return "low"

def draw_overlay(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> str:
    overlay = img_bgr.copy()
    for it in items:
        x1,y1,x2,y2 = it["bounding_box"]
        label = f"{it['damage_type']} • {it['severity']}"
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(overlay,label,(x1,max(10,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buf.tobytes()).decode("ascii")

@app.get("/health")
def health(): return {"ok": True}

@app.get("/ready")
def ready():
    img = np.zeros((64,64,3), dtype=np.uint8)
    _ = model.predict(img, conf=0.1, verbose=False)
    return {"ready": True, "model": MODEL_VERSION}

@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "thresholds": {
            "conf_dent": CONF_DENT, "conf_scratch": CONF_SCRATCH, "conf_generic": CONF_GENERIC,
            "sev_med": SEV_MED, "sev_high": SEV_HIGH
        },
        "quality_gates": {
            "min_short_side": MIN_SHORT_SIDE, "min_blur_var": MIN_BLUR_VAR,
            "bright_min": BRIGHT_MIN, "bright_max": BRIGHT_MAX
        }
    }

def _analyze(image: Image.Image, want_overlay: bool) -> Dict[str, Any]:
    rgb = np.array(image); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    warnings = assess_quality(bgr)

    results = model.predict(image, conf=min(CONF_GENERIC, CONF_DENT, CONF_SCRATCH), verbose=False)
    boxes = results[0].boxes

    findings: List[Dict[str, Any]] = []
    # sort boxes for deterministic output
    order = []
    for i, b in enumerate(boxes):
        cls = int(b.cls.cpu().numpy()[0])
        conf = float(b.conf.cpu().numpy()[0])
        x1,y1,x2,y2 = map(int, b.xyxy.cpu().numpy()[0])
        order.append((LABELS[cls] if cls < len(LABELS) else "zzz", -conf, x1, i))
    for _,__,___,i in sorted(order):
        b = boxes[i]
        cls = int(b.cls.cpu().numpy()[0])
        if cls >= len(LABELS): continue
        label = LABELS[cls]
        conf = float(b.conf.cpu().numpy()[0])
        # per-class confidence gate
        if conf < class_conf(label): continue
        x1,y1,x2,y2 = map(int, b.xyxy.cpu().numpy()[0])
        bw,bh = x2-x1, y2-y1
        pixel_area = max(1, bw*bh)
        area_ratio = pixel_area / float(w*h)
        sev = severity_from_area_ratio(area_ratio, conf)
        location = side_label_from_box(w, h, x1, y1, x2, y2, label)

        findings.append({
            "part": location,
            "damage_type": label,
            "severity": sev,
            "estimated_surface": f"{(area_ratio*100):.2f} % of image",
            "confidence": round(conf,3),
            "bounding_box": [x1,y1,x2,y2]
        })

    next_best_action: Optional[str] = None
    if not findings:
        next_best_action = "No clear damage found. Capture a wider shot with full car and good lighting; add a second 3/4 angle."

    resp: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "summary": f"{len(findings)} damage regions detected.",
        "findings": findings,
        "model_version": MODEL_VERSION,
        "warnings": warnings,
        "next_best_action": next_best_action
    }
    if want_overlay and findings:
        resp["overlay_b64"] = draw_overlay(bgr, findings)
    return resp

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), overlay: bool = Query(OVERLAY_DEFAULT)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")
    return JSONResponse(content=_analyze(image, want_overlay=overlay))

# Back-compat: your legacy shape
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")
    full = _analyze(image, want_overlay=False)
    findings = full["findings"]
    if not findings: return JSONResponse(content={"message": "No damage detected in the image."})
    legacy=[]
    for f in findings:
        legacy.append({
            "part": f["part"], "damage_type": f["damage_type"],
            "severity": f["severity"], "estimated_surface": f["estimated_surface"],
            "confidence": f"{float(f['confidence']):.1%}", "bounding_box": f["bounding_box"]
        })
    return JSONResponse(content=legacy)
