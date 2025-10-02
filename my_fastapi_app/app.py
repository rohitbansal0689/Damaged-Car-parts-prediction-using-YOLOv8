# my_fastapi_app/app.py
import os, io, uuid, pathlib, random, base64
from typing import List, Dict, Any, Optional
import numpy as np, cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

APP_VERSION = "1.6.5"
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.getenv("MODEL_PATH", str(SCRIPT_DIR / "best.pt"))

# -------- Minimal env (keep it simple)
CONF_MIN_GATE   = float(os.getenv("CONF_MIN_GATE", "0.25"))  # first gate to YOLO
CONF_DENT       = float(os.getenv("CONF_DENT", "0.70"))
CONF_SCRATCH    = float(os.getenv("CONF_SCRATCH", "0.35"))
CONF_GENERIC    = float(os.getenv("CONF_GENERIC", "0.25"))

# Windshield-specific & shape guards (to prevent thin top-strip false positives)
CONF_WINDSHIELD   = float(os.getenv("CONF_WINDSHIELD", "0.80"))  # stricter than generic
MIN_WSHIELD_HF    = float(os.getenv("MIN_WSHIELD_HF", "0.18"))   # min windshield box height as fraction of image height
BORDER_THIN_FRAC  = float(os.getenv("BORDER_THIN_FRAC", "0.12")) # drop border-hugging bands thinner than this
BORDER_PIX        = int(os.getenv("BORDER_PIX", "3"))            # “within N px of an edge”

OVERLAY_DEFAULT = os.getenv("OVERLAY_DEFAULT", "0") == "1"
IMG_SIZE = int(os.getenv("IMG_SIZE", "960"))
AGNOSTIC_NMS = os.getenv("AGNOSTIC_NMS", "0") == "1"

# Light quality checks (messages only; never block)
MIN_SHORT_SIDE  = int(os.getenv("MIN_SHORT_SIDE", "640"))
MIN_BLUR_VAR    = float(os.getenv("MIN_BLUR_VAR", "60"))
BRIGHT_MIN      = float(os.getenv("BRIGHT_MIN", "60"))
BRIGHT_MAX      = float(os.getenv("BRIGHT_MAX", "200"))

# Optional lightweight auth
INTERNAL_TOKEN  = os.getenv("INTERNAL_TOKEN", "")

# Constant for surface estimate + severity (kept simple)
DM2_DIVISOR = 500.0
DM2_MED     = 2.0
DM2_HIGH    = 5.0

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

# ---- App ----
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "0") == "1"
app = FastAPI(
    title="Vehicle Damage Assessment API",
    description="Detect and classify car damage from images.",
    version=APP_VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
)

# ---- Model ----
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model from '{MODEL_PATH}': {e}")

# ---- Helpers ----
def require_token(request: Request):
    if INTERNAL_TOKEN and request.headers.get("x-internal-token") != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

def normalize_label(lbl: str) -> str:
    """Normalize minor spelling/spacing differences in labels for stable API output."""
    l = (lbl or "").lower().replace("-", " ").strip()
    if l in ("damaged wind shield", "damaged windshield", "damaged windscreen"):
        return "damaged windshield"
    return lbl

def class_conf(lbl: str) -> float:
    l = (lbl or "").lower()
    l_ns = l.replace(" ", "").replace("-", "")
    if l == "dent": return CONF_DENT
    if l == "scratch": return CONF_SCRATCH
    if l_ns in ("damagedwindshield", "windshield"):
        return CONF_WINDSHIELD
    return CONF_GENERIC

def assess_quality(img_bgr: np.ndarray) -> List[str]:
    warnings = []
    h, w = img_bgr.shape[:2]
    if MIN_SHORT_SIDE > 0 and min(h, w) < MIN_SHORT_SIDE:
        warnings.append(f"Image is small ({w}x{h}); use ≥{MIN_SHORT_SIDE}px short side.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < BRIGHT_MIN: warnings.append("Image too dark; shoot in better light.")
    if gray.mean() > BRIGHT_MAX: warnings.append("Image too bright; reduce glare/exposure.")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR_VAR:
        warnings.append("Image looks blurry; hold steady or move closer.")
    return warnings

def side_label_from_box(img_w:int, img_h:int, x1:int, y1:int, x2:int, y2:int, label:Optional[str]=None) -> str:
    """Door & dent share door-region wording; windshield special; others fall back to thirds."""
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    l = (label or "").lower().replace("-", " ")
    if ("door" in l) or (l == "dent"):
        return "front door area" if cx < (img_w * 0.5) else "rear door area"
    if "windshield" in l or "wind shield" in l or "windscreen" in l:
        return "windshield area"
    horiz = "left side" if cx < (img_w/3.0) else "right side" if cx > (2.0*img_w/3.0) else "center"
    vert  = "front area" if cy < (img_h/3.0) else "rear area"  if cy > (img_h * 2.0 / 3.0) else "center"
    return vert if vert != "center" else (horiz if horiz != "center" else "side area")

def severity_from_dm2(dm2: float, conf: float) -> str:
    if dm2 >= DM2_HIGH or (dm2 >= DM2_HIGH*0.6 and conf >= 0.75): return "severe"
    if dm2 >= DM2_MED: return "moderate"
    return "low"

def draw_overlay(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> str:
    overlay = img_bgr.copy()
    for it in items:
        x1, y1, x2, y2 = it["bounding_box"]
        label = f"{it['damage_type']} - {it['severity']}"
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(overlay, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buf.tobytes()).decode("ascii")

def conf_to_num(c):
    if isinstance(c, str) and c.endswith("%"):
        try: return float(c[:-1]) / 100.0
        except: return 0.0
    try: return float(c)
    except: return 0.0

def dedupe_by_label_and_part(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep top-1 per (damage_type, part). Different labels on same part both remain."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for it in items:
        key = (it["damage_type"], it["part"])
        groups.setdefault(key, []).append(it)
    kept: List[Dict[str, Any]] = []
    for _, grp in groups.items():
        grp.sort(key=lambda d: conf_to_num(d.get("confidence", 0)), reverse=True)
        kept.append(grp[0])
    return kept

# --- shape/position guard to kill thin border bands & implausible windshields
def passes_shape_guards(lbl: str, x1:int, y1:int, x2:int, y2:int, img_w:int, img_h:int) -> bool:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    hf = h / float(img_h)
    lbl_ns = (lbl or "").lower().replace(" ", "").replace("-", "")

    # reject ultra-thin bands stuck to top/bottom edges (common false positives)
    if hf < BORDER_THIN_FRAC and (y1 <= BORDER_PIX or y2 >= img_h - BORDER_PIX):
        return False

    # windshield-specific: must be a sufficiently tall patch
    if lbl_ns in ("damagedwindshield", "windshield"):
        if hf < MIN_WSHIELD_HF:
            return False

    return True

# ---- Endpoints ----
@app.get("/health")
def health(): return {"ok": True}

@app.get("/ready")
def ready():
    img = np.zeros((64,64,3), dtype=np.uint8)
    _ = model.predict(img, conf=0.1, verbose=False)
    return {"ready": True, "model_path": MODEL_PATH}

@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "model_path": MODEL_PATH,
        "thresholds": {
            "conf_min_gate": CONF_MIN_GATE,
            "conf_dent": CONF_DENT,
            "conf_scratch": CONF_SCRATCH,
            "conf_generic": CONF_GENERIC,
            "conf_windshield": CONF_WINDSHIELD,
        },
        "quality": {
            "min_short_side": MIN_SHORT_SIDE,
            "min_blur_var": MIN_BLUR_VAR,
            "bright_min": BRIGHT_MIN,
            "bright_max": BRIGHT_MAX,
        },
        "guards": {
            "min_windshield_height_frac": MIN_WSHIELD_HF,
            "border_thin_frac": BORDER_THIN_FRAC,
            "border_px": BORDER_PIX,
        }
    }

def _analyze(image: Image.Image, want_overlay: bool) -> Dict[str, Any]:
    rgb = np.array(image); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    warnings = assess_quality(bgr)

    res = model.predict(
        image,
        conf=CONF_MIN_GATE,
        imgsz=IMG_SIZE,
        agnostic_nms=AGNOSTIC_NMS,
        verbose=False
    )[0]
    names = getattr(res, "names", None) or getattr(model, "names", None) or {}
    boxes = getattr(res, "boxes", [])

    raw: List[Dict[str, Any]] = []
    # stable order: label asc, conf desc, x1 asc
    order = []
    for i, b in enumerate(boxes):
        cls_i = int(b.cls.cpu().numpy()[0])
        conf  = float(b.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
        lbl = names.get(cls_i, str(cls_i))
        order.append((lbl, -conf, x1, i))

    for _,__,___, i in sorted(order):
        b = boxes[i]
        cls_i = int(b.cls.cpu().numpy()[0])
        conf  = float(b.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])

        lbl_raw = names.get(cls_i, str(cls_i))
        lbl = normalize_label(lbl_raw)

        # per-class threshold
        if conf < class_conf(lbl):
            continue

        # shape/position guard
        if not passes_shape_guards(lbl, x1, y1, x2, y2, w, h):
            continue

        area_px = max(1, (x2 - x1) * (y2 - y1))
        dm2 = round(area_px / DM2_DIVISOR, 2)

        part = side_label_from_box(w, h, x1, y1, x2, y2, lbl)
        raw.append({
            "part": part,
            "damage_type": lbl,
            "severity": severity_from_dm2(dm2, conf),
            "estimated_surface": f"{dm2} dm²",
            "confidence": f"{conf:.1%}",
            "bounding_box": [x1, y1, x2, y2]
        })

    # strict dedupe within same (label, part); different labels on same part both remain
    findings = dedupe_by_label_and_part(raw)

    resp: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "summary": f"{len(findings)} damage regions detected.",
        "findings": findings,
        "warnings": warnings,
        "next_best_action": None if findings else "No clear damage found. Capture a wider shot with full car and good lighting; add a second 3/4 angle."
    }
    if want_overlay and findings:
        resp["overlay_b64"] = draw_overlay(bgr, findings)
    return resp

@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...), overlay: bool = Query(OVERLAY_DEFAULT)):
    require_token(request)
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")
    return JSONResponse(content=_analyze(image, want_overlay=overlay))

# legacy
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    require_token(request)
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    out = _analyze(image, want_overlay=False)
    return JSONResponse(content=out.get("findings", []) or {"message": "No damage detected in the image."})
