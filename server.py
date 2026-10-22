"""PlantDoc web server: JSON API + static frontend (web/).

Run locally:  uvicorn server:app --reload
Deploy:       see Dockerfile (listens on $PORT)
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # iPhone photos are HEIC by default
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

from src import accounts
from src.accounts import AuthError
from src.agri_advisory import get_advisory
from src.expert_system import META, build_plan
from src.fuzzy_engine import evaluate_trust
from src.gradcam import GradCAM
from src.i18n import LANGUAGES, t, t_disease
from src.model import PlantDocModel
from src.quality_gate import check_quality

WEB_DIR = Path(__file__).parent / "web"
MAX_SIDE = 2048
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

state: dict = {}
infer_lock = threading.Lock()  # the model + Grad-CAM hooks are shared; serialise inference


@asynccontextmanager
async def lifespan(_: FastAPI):
    accounts.init_db()
    state["model"] = PlantDocModel()
    yield


app = FastAPI(title="PlantDoc Leaf Doctor", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)

COOKIE = "pd_session"
_attempts: dict[str, deque] = defaultdict(deque)


@app.exception_handler(AuthError)
async def _auth_error(_: Request, e: AuthError):
    return JSONResponse({"detail": e.message}, status_code=e.status)


@app.middleware("http")
async def csrf_and_headers(request: Request, call_next):
    # Cookie auth is SameSite=Lax; additionally require a custom header on writes,
    # which a cross-site form post cannot set.
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/") \
            and request.headers.get("x-requested-with") != "PlantDoc":
        return JSONResponse({"detail": "Missing X-Requested-With header."}, status_code=403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    return resp


def current_user(request: Request) -> dict | None:
    return accounts.user_for_token(request.cookies.get(COOKIE))


def require_user(user: dict | None = Depends(current_user)) -> dict:
    if not user:
        raise HTTPException(401, "Please sign in.")
    return user


def _set_cookie(request: Request, response: Response, token: str) -> None:
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, token, max_age=accounts.SESSION_TTL, httponly=True, samesite="lax",
                        secure=secure, path="/")


def _throttle_key(request: Request, email: str) -> str:
    return f"{request.client.host if request.client else '?'}|{email.strip().lower()}"


def _throttle(request: Request, email: str) -> None:
    q, now = _attempts[_throttle_key(request, email)], time.time()
    while q and now - q[0] > 300:
        q.popleft()
    if len(q) >= 8:
        raise HTTPException(429, "Too many attempts. Please wait a few minutes and try again.")
    q.append(now)


def _png_b64(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _quality_tip(memberships: dict, lang: str) -> str:
    # Same priority as quality_gate._retake_tip (lighting first)
    if memberships["brightness"]["dark"] > 0.3:
        return t("tip_dark", lang)
    if memberships["brightness"]["over"] > 0.3:
        return t("tip_overexposed", lang)
    tips = []
    if max(memberships["blur"]["blurry"], memberships["blur"]["streaked"]) > 0.3:
        tips.append(t("tip_blurry", lang))
    if memberships["coverage"]["low"] > 0.3:
        tips.append(t("tip_low_coverage", lang))
    return " ".join(tips) or t("tip_borderline", lang)


@app.get("/api/health")
def health():
    model = state["model"]
    return {"ok": True, "demo": model.is_demo, "weights": model.weights_source,
            "trust_threshold": model.trust_threshold}


def _thumb(image: Image.Image) -> str:
    im = image.copy()
    im.thumbnail((200, 200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_ctx(raw: str | None) -> dict:
    try:
        v = json.loads(raw) if raw else {}
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _preds_json(preds, lang):
    return [{"name": n, "label": t_disease(n, lang), "confidence": p,
             "pathogen": (get_advisory(n) or {}).get("pathogen")} for n, p in preds]


@app.post("/api/diagnose")
def diagnose(file: UploadFile = File(...), lang: str = Form("en"), context: str = Form(""), save: str = Form("1"),
             user: dict | None = Depends(current_user)):
    if lang not in LANGUAGES:
        lang = "en"
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is larger than 15 MB.")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Could not read this file as an image. Please use a JPG, PNG, WEBP or HEIC photo.")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    profile = user["profile"] if user else None
    ctx_in = _parse_ctx(context)

    model: PlantDocModel = state["model"]
    with infer_lock:
        passed, q_score, _ = check_quality(image)
        features = check_quality.last["features"]
        memberships = check_quality.last["memberships"]
        result: dict = {"quality": {"score": q_score, "passed": passed, "features": features},
                        "demo": model.is_demo, "threshold": model.trust_threshold}

        if not passed:
            return {**result, "status": "REJECTED_QUALITY", "tip": _quality_tip(memberships, lang)}

        preds, similarity, is_ood = model.diagnose(image, top_k=3)
        if is_ood:
            return {**result, "status": "REJECTED_OOD", "message": t("ood_rejected", lang),
                    "similarity": similarity}

        top1_name, top1_conf = preds[0]
        trust = evaluate_trust(q_score, top1_conf, threshold=model.trust_threshold)
        result["trust"] = trust.trust
        result["predictions"] = _preds_json(preds, lang)
        uncertain = trust.status == "UNCERTAIN"
        gradcam = None
        if not uncertain:
            try:
                _, blended = GradCAM(model.model).generate(image)
                gradcam = _png_b64(blended)
            except Exception:
                pass

    brown = features.get("brown_ratio")
    plan = build_plan(None if uncertain else top1_name, lang, confidence=top1_conf, trust=trust.trust,
                      brown_ratio=brown, uncertain=uncertain, ctx=ctx_in, profile=profile,
                      candidates=[t_disease(n, lang) for n, _ in preds[:2]])
    out = {**result, "status": "UNCERTAIN" if uncertain else "TRUSTED", "plan": plan,
           "brown_ratio": brown, "gradcam": gradcam}
    if uncertain:
        out.update(message=t("uncertain_warning", lang),
                   reason=t("uncertain_reason", lang, trust=trust.trust, threshold=model.trust_threshold))
    if user and save != "0":
        out["history_id"] = accounts.add_history(
            user["id"], status=out["status"], disease=None if uncertain else top1_name,
            confidence=top1_conf, trust=trust.trust, quality=q_score, thumb=_thumb(image),
            ctx={"ctx": ctx_in, "brown_ratio": brown, "uncertain": uncertain,
                 "preds": [[n, p] for n, p in preds]})
    return out


@app.post("/api/advisor")
def advisor(body: dict = Body(...), user: dict | None = Depends(current_user)):
    """Expert recommendation without a new photo: used by the Advisor tab and to refine a result."""
    lang = body.get("lang") if body.get("lang") in LANGUAGES else "en"
    disease = body.get("disease")
    uncertain = bool(body.get("uncertain"))
    if not uncertain and disease not in META:
        raise HTTPException(400, "Unknown disease.")
    brown = body.get("brown_ratio")
    brown = float(brown) if isinstance(brown, (int, float)) else None
    cands = [t_disease(n, lang) for n in body.get("candidates", []) if n in META][:2]
    plan = build_plan(None if uncertain else disease, lang, brown_ratio=brown, uncertain=uncertain,
                      candidates=cands, ctx=body.get("ctx") if isinstance(body.get("ctx"), dict) else {},
                      profile=user["profile"] if user else None)
    return {"plan": plan}


@app.get("/api/diseases")
def diseases(lang: str = "en"):
    lang = lang if lang in LANGUAGES else "en"
    return [{"name": n, "label": t_disease(n, lang), "crop": m["crop"]} for n, m in META.items()]


# ---- accounts ----------------------------------------------------------------
@app.get("/api/auth/me")
def me(user: dict | None = Depends(current_user)):
    return {"user": user}


@app.post("/api/auth/register")
def register(request: Request, response: Response, body: dict = Body(...)):
    email = str(body.get("email", ""))
    _throttle(request, email)
    uid = accounts.register(email, str(body.get("password", "")), str(body.get("name", "")))
    accounts.update_profile(uid, {"language": body.get("language", "en")})
    _set_cookie(request, response, accounts.create_session(uid))
    return {"user": accounts.get_user(uid)}


@app.post("/api/auth/login")
def login(request: Request, response: Response, body: dict = Body(...)):
    email = str(body.get("email", ""))
    _throttle(request, email)
    uid = accounts.login(email, str(body.get("password", "")))
    _attempts.pop(_throttle_key(request, email), None)
    _set_cookie(request, response, accounts.create_session(uid))
    return {"user": accounts.get_user(uid)}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    accounts.end_session(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.patch("/api/profile")
def patch_profile(body: dict = Body(...), user: dict = Depends(require_user)):
    return {"user": accounts.update_profile(user["id"], body)}


@app.post("/api/auth/password")
def password(request: Request, body: dict = Body(...), user: dict = Depends(require_user)):
    th = accounts._th(request.cookies.get(COOKIE, ""))
    accounts.change_password(user["id"], str(body.get("current", "")), str(body.get("new", "")), th)
    return {"ok": True}


@app.post("/api/account/delete")
def delete_account(response: Response, body: dict = Body(...), user: dict = Depends(require_user)):
    accounts.delete_account(user["id"], str(body.get("password", "")))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


# ---- history -------------------------------------------------------------------
@app.get("/api/history")
def history(lang: str = "en", user: dict = Depends(require_user)):
    lang = lang if lang in LANGUAGES else "en"
    items = accounts.list_history(user["id"])
    for it in items:
        it["label"] = t_disease(it["disease"], lang) if it["disease"] else None
    return {"items": items}


@app.get("/api/history/{hid}")
def history_item(hid: int, lang: str = "en", user: dict = Depends(require_user)):
    lang = lang if lang in LANGUAGES else "en"
    it = accounts.get_history(user["id"], hid)
    if not it:
        raise HTTPException(404, "Not found.")
    c = it["ctx"]
    preds = c.get("preds", [])
    plan = build_plan(it["disease"], lang, confidence=it["confidence"], trust=it["trust"],
                      brown_ratio=c.get("brown_ratio"), uncertain=bool(c.get("uncertain")),
                      candidates=[t_disease(n, lang) for n, _ in preds[:2]],
                      ctx=c.get("ctx"), profile=user["profile"])
    return {**it, "predictions": _preds_json(preds, lang), "plan": plan, "brown_ratio": c.get("brown_ratio")}


@app.delete("/api/history/{hid}")
def history_delete(hid: int, user: dict = Depends(require_user)):
    accounts.delete_history(user["id"], hid)
    return {"ok": True}


@app.delete("/api/history")
def history_clear(user: dict = Depends(require_user)):
    accounts.delete_history(user["id"])
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
