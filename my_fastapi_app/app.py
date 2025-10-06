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

APP_VERSION = "1.6.7"
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Default to trained.pt; override via MODEL_PATH env
MODEL_PATH = os.getenv("MODEL_PATH", str(SCRIPT_DIR / "trained.pt"))

# -------- Thresholds (env-tunable) --------
CONF_MIN_GATE   = float(os.getenv("CONF_MIN_GATE", "0.25"))  # forward pass gate
CONF_DENT       = float(os.getenv("CONF_DENT", "0.70"))
CONF_SCRATCH    = float(os.getenv("CONF_SCRATCH", "0.35"))
CONF_GENERIC    = float(os.getenv("CONF_GENERIC", "0.25"))

# Windshield-specific & shape guards
CONF_WINDSHIELD   = float(os.getenv("CONF_WINDSHIELD", "0.80"))
MIN_WSHIELD_HF    = float(os.getenv("MIN_WSHIELD_HF", "0.18"))   # min windshield box height as frac of image height
BORDER_THIN_FRAC  = float(os.getenv("BORDER_THIN_FRAC", "0.12")) # reject ultra-thin bands at top/bottom edges
BORDER_PIX        = int(os.getenv("BORDER_PIX", "3"))            # “within N px of an edge”

# Trained model includes this class
CONF_FLAT_TIRE    = float(os.getenv("CONF_FLAT_TIRE", "0.80"))

OVERLAY_DEFAULT = os.getenv("OVERLAY_DEFAULT", "0") == "1"
IMG_SIZE = int(os.getenv("IMG_SIZE", "960"))
AGNOSTIC_NMS = os.getenv("AGNOSTIC_NMS", "0") == "1"

# Light-quality hints (non-blocking)
MIN_SHORT_SIDE  = int(os.getenv("MIN_SHORT_SIDE", "640"))
MIN_BLUR_VAR    = float(os.getenv("MIN_BLUR_VAR", "60"))
BRIGHT_MIN      = float(os.getenv("BRIGHT_MIN", "60"))
BRIGHT_MAX      = float(os.getenv("BRIGHT_MAX", "200"))

# Optional lightweight auth
INTERNAL_TOKEN  = os.getenv("INTERNAL_TOKEN", "")

# Surface estimate + severity buckets
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
    title="Konstrukta inspection API",
    version=APP_VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_tags=[
            {"name": "Damage Analysis", "description": "Upload an image to analyze vehicle damage"}],
   swagger_ui_parameters={
         "defaultModelsExpandDepth": -1,  # HIDE the entire “Schemas” section
         "defaultModelExpandDepth": 1,    # (optional) don’t auto-expand single models
     }
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
    """
    Normalize minor spelling/spacing/underscore differences for stable output.
    Class set in trained.pt: {dent, scratch, crack, shattered_glass, broken_lamp, flat_tire}
    We map 'crack' & 'shattered_glass' to a temporary placeholder 'glass damage'
    and disambiguate to 'damaged window' vs 'damaged windshield' later using box geometry.
    """
    l = (lbl or "").lower().replace("-", " ").replace("_", " ").strip()
    if l in {"shattered glass", "cracked glass"}:
        return "glass damage"
    if l == "crack":
        return "glass damage"  # treat as glass-like for MVP
    if l in {"flat tire", "flat tyre"}:
        return "flat tire"
    if l == "broken lamp":
        return "broken lamp"
    if l in {
        "damaged door","damaged bumper","damaged hood","damaged fender",
        "damaged mirror","damaged headlight","damaged window","damage","panel damage"
    }:
        return "dent"
    return lbl or ""

def class_conf(lbl: str) -> float:
    l = (lbl or "").lower()
    l_ns = l.replace(" ", "").replace("-", "").replace("_", "")
    if l == "dent": return CONF_DENT
    if l == "scratch": return CONF_SCRATCH
    if l_ns in ("damagedwindshield", "windshield"): return CONF_WINDSHIELD
    if l_ns in ("flattire", "tireflat"): return CONF_FLAT_TIRE
    # 'damaged window', 'broken lamp', and fallback labels use generic
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
    """
    Heuristics for 'part' strings.
    - Door & dent share door-region wording.
    - Windshield explicit.
    - Flat tire → left/right wheel area.
    - Broken lamp → left/right headlight area.
    - Otherwise thirds-based fallback (front/center/rear vs left/right).
    """
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    l = (label or "").lower().replace("-", " ").replace("_", " ")
    if ("door" in l) or (l == "dent"):
        return "front door area" if cx < (img_w * 0.5) else "rear door area"
    if "windshield" in l or "wind shield" in l or "windscreen" in l:
        return "windshield area"
    if "damaged window" in l:
        return "left window area" if cx < img_w/2 else "right window area"
    if "flat" in l and ("tire" in l or "tyre" in l):
        return "left wheel area" if cx < img_w/2 else "right wheel area"
    if "lamp" in l or "headlight" in l:
        return "left headlight area" if cx < (img_w * 0.5) else "right headlight area"
    horiz = "left side" if cx < (img_w/3.0) else "right side" if cx > (2.0*img_w/3.0) else "center"
    vert  = "front area" if cy < (img_h/3.0) else "rear area"  if cy > (img_h * 2.0 / 3.0) else "center"
    return vert if vert != "center" else (horiz if horiz != "center" else "side area")

def severity_from_dm2(dm2: float, conf: float) -> str:
    if dm2 >= DM2_HIGH or (dm2 >= DM2_HIGH*0.6 and conf >= 0.75): return "severe"
    if dm2 >= DM2_MED: return "moderate"
    return "low"

# def draw_overlay(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> str:
#     overlay = img_bgr.copy()
#     for it in items:
#         x1, y1, x2, y2 = it["bounding_box"]
#         label = f"{it['damage_type']} - {it['severity']}"
#         cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
#         cv2.putText(overlay, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
#     _, buf = cv2.imencode(".jpg", overlay)
#     return base64.b64encode(buf.tobytes()).decode("ascii")

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

def thin_border_guard(y1:int, y2:int, img_h:int) -> bool:
    """Reject ultra-thin horizontal bands at the very top/bottom (common false positives)."""
    h = max(1, y2 - y1)
    hf = h / float(img_h)
    if hf < BORDER_THIN_FRAC and (y1 <= BORDER_PIX or y2 >= img_h - BORDER_PIX):
        return False
    return True

def finalize_glass_label(lbl_norm: str, x1:int, y1:int, x2:int, y2:int, img_w:int, img_h:int) -> str:
    """
    If label is 'glass damage', decide 'damaged window' vs 'damaged windshield'
    via simple geometry:
      - windows: smaller area + off-center horizontally
      - windshield: larger/central span
    Tunable knobs: AREA_FRAC_MAX (default 0.25), CENTER_BAND (0.40–0.60)
    """
    if lbl_norm != "glass damage":
        return lbl_norm
    area_frac = max(1, (x2 - x1) * (y2 - y1)) / float(img_w * img_h)
    cx = (x1 + x2) / 2.0 / img_w
    AREA_FRAC_MAX = 0.25
    LEFT_EDGE, RIGHT_EDGE = 0.35, 0.65
    if area_frac < AREA_FRAC_MAX and (cx < LEFT_EDGE or cx > RIGHT_EDGE):
        return "damaged window"
    return "damaged windshield"

def windshield_height_guard(final_label: str, y1:int, y2:int, img_h:int) -> bool:
    """Apply windshield min-height guard only when final label is windshield."""
    if final_label != "damaged windshield":
        return True
    h = max(1, y2 - y1)
    hf = h / float(img_h)
    return hf >= MIN_WSHIELD_HF

# ---- Endpoints ----
# @app.get("/health")
# def health(): return {"ok": false}
#
# @app.get("/ready")
# def ready():
#     img = np.zeros((64,64,3), dtype=np.uint8)
#     _ = model.predict(img, conf=0.1, verbose=False)
#     return {"ready": True, "model_path": MODEL_PATH}
#
# @app.get("/version")
# def version():
#     return {
#         "app_version": APP_VERSION,
#         "model_path": MODEL_PATH,
#         "thresholds": {
#             "conf_min_gate": CONF_MIN_GATE,
#             "conf_dent": CONF_DENT,
#             "conf_scratch": CONF_SCRATCH,
#             "conf_generic": CONF_GENERIC,
#             "conf_windshield": CONF_WINDSHIELD,
#             "conf_flat_tire": CONF_FLAT_TIRE,
#         },
#         "quality": {
#             "min_short_side": MIN_SHORT_SIDE,
#             "min_blur_var": MIN_BLUR_VAR,
#             "bright_min": BRIGHT_MIN,
#             "bright_max": BRIGHT_MAX,
#         },
#         "guards": {
#             "min_windshield_height_frac": MIN_WSHIELD_HF,
#             "border_thin_frac": BORDER_THIN_FRAC,
#             "border_px": BORDER_PIX,
#         }
#     }

def _analyze(image: Image.Image) -> Dict[str, Any]:
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
        lbl_norm = normalize_label(lbl_raw)

        # first generic border-thin guard
        if not thin_border_guard(y1, y2, h):
            continue

        # for glass-like labels, decide window vs windshield now
        final_label = finalize_glass_label(lbl_norm, x1, y1, x2, y2, w, h)

        # per-class gate using the final label
        if conf < class_conf(final_label):
            continue

        # apply windshield height guard only if final label is windshield
        if not windshield_height_guard(final_label, y1, y2, h):
            continue

        area_px = max(1, (x2 - x1) * (y2 - y1))
        dm2 = round(area_px / DM2_DIVISOR, 2)

        part = side_label_from_box(w, h, x1, y1, x2, y2, final_label)

        severity = severity_from_dm2(dm2, conf)
        if final_label == "flat tire":
            severity = "severe"  # business rule: a flat is always severe

        raw.append({
            "part": part,
            "damage_type": final_label,
            "severity": severity,
            "estimated_surface": f"{dm2} dm²",
            "confidence": f"{conf:.1%}",
            "bounding_box": [x1, y1, x2, y2]
        })

    findings = dedupe_by_label_and_part(raw)

    resp: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "summary": f"{len(findings)} damage regions detected.",
        "findings": findings,
        "warnings": warnings,
        "next_best_action": None if findings else "No clear damage found. Capture a wider shot with full car and good lighting; add a second 3/4 angle."
    }
    return resp


@app.post("/analyze", tags=["Damage Analysis"])
async def analyze(request: Request, file: UploadFile = File(...)):
    require_token(request)
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {e}")
    return JSONResponse(content=_analyze(image))

# --- Legacy endpoint (kept here but hidden from docs) ---
# @app.post("/predict")
# async def predict(request: Request, file: UploadFile = File(...)):
#     require_token(request)
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     out = _analyze(image, want_overlay=False)
#     return JSONResponse(content=out.get("findings", []) or {"message": "No damage detected in the image."})
