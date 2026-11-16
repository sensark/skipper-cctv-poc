from __future__ import annotations

import base64
import hashlib
import os
import shutil
import tempfile
import time
import urllib.request
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

load_dotenv(Path(__file__).parent / ".env")

warnings.filterwarnings("ignore", message=".*legacy channel.*")

DEPLOYMENT_URL = "https://serverless.roboflow.com"
WORKSPACE = "sensark"
WORKFLOW = (
    "construction-site-po4lz-1-rfdetr-small-t1-"
    "vconstruction-site-po4lz-1-rfdetr-small-t1-logic"
)
INFERENCE_WIDTH = 1280
MAX_IN_FLIGHT = 4
PPE_MODEL = "Trained model"
YOLO_BASELINE_MODEL = "Baseline model"
YOLO_WEIGHTS_PATH = Path(__file__).parent / "models" / "yolo26n.pt"
YOLO_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
)
YOLO_WEIGHTS_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
YOLO_INFERENCE_LOCK = Lock()

VIOLATIONS = {
    "without_helmet": "No Helmet",
    "without_vest": "No Safety Vest",
}
COMPLIANT = {"helmet", "vest"}
COLORS_BGR = {
    "without_helmet": (68, 68, 239),
    "without_vest": (22, 115, 249),
    "helmet": (129, 185, 16),
    "vest": (129, 185, 16),
}
SEVERITY = {"No Helmet": "critical", "No Safety Vest": "high"}

st.set_page_config(page_title="Skipper | PPE Monitoring", page_icon="🦺", layout="wide")

st.markdown(
    """
<style>
:root { color-scheme:dark; --bg:#080d14; --surface:#101923; --surface-hi:#16232e; --line:#253440;
        --txt:#edf4f2; --mute:#91a19f; --lime:#c8f169; --cyan:#63d9d2;
        --red:#ff6677; --orange:#ffb454; --green:#c8f169; }
.stApp { background:radial-gradient(900px 500px at 8% -10%,#18312f 0%,transparent 68%),
         radial-gradient(750px 450px at 100% 0%,#18243a 0%,transparent 62%),var(--bg); color:var(--txt); }
header[data-testid="stHeader"] { background:transparent; }
section[data-testid="stSidebar"] { background:rgba(12,19,27,.94); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color:var(--lime); letter-spacing:.02em; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
.brand { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.15rem;
         padding:0 0 1.2rem; border-bottom:1px solid var(--line); }
.brand-lockup { display:flex; align-items:center; gap:13px; }
.brand-mark { display:grid; place-items:center; width:44px; height:44px; border-radius:14px;
              color:#10170a; font-size:1.3rem; font-weight:950;
              background:linear-gradient(145deg,#e1ff9d,var(--lime)); box-shadow:0 0 28px #c8f16930; }
.brand h1 { font-size:1.75rem; margin:0; font-weight:850; letter-spacing:-.04em; color:var(--txt); }
.brand small { color:var(--mute); display:block; margin-top:3px; font-size:.78rem; letter-spacing:.04em; }
.pill { padding:7px 13px; border-radius:999px; font-weight:800; font-size:.7rem; letter-spacing:.11em;
        border:1px solid var(--line); background:var(--surface); color:var(--mute); }
.pill.live { color:#fff; background:#ff66771a; border-color:#ff667766; }
.pill.live::before { content:""; display:inline-block; width:8px; height:8px; margin-right:8px;
                     border-radius:50%; background:var(--red); animation:pulse 1.5s infinite; }
.pill.done { color:var(--lime); border-color:#c8f16966; background:#c8f16912; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 #ff667788} 70%{box-shadow:0 0 0 8px #ff667700} 100%{box-shadow:0 0 0 0 #ff667700} }
.panel-title { display:flex; align-items:center; gap:9px; color:var(--mute); text-transform:uppercase;
               letter-spacing:.13em; font-size:.68rem; font-weight:800; margin:4px 0 10px; }
.panel-title::before { content:""; width:7px; height:7px; border-radius:2px; background:var(--lime);
                       box-shadow:0 0 12px #c8f16966; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:9px; margin-bottom:14px; }
.kpi { position:relative; overflow:hidden; min-height:86px; background:linear-gradient(145deg,#15212a,#101820);
       border:1px solid var(--line); border-radius:14px; padding:13px 14px; transition:transform .18s ease,border-color .18s ease; }
.kpi:hover { transform:translateY(-2px); border-color:#53674c; }
.kpi::after { content:""; position:absolute; width:52px; height:52px; right:-24px; top:-25px;
              border-radius:50%; background:var(--lime); opacity:.08; filter:blur(2px); }
.kpi .l { color:var(--mute); font-size:.61rem; line-height:1.35; text-transform:uppercase; letter-spacing:.08em; }
.kpi .v { margin-top:6px; font-size:1.75rem; font-weight:850; line-height:1; letter-spacing:-.04em; }
.kpi.red .v { color:var(--red); } .kpi.orange .v { color:var(--orange); }
.kpi.cyan .v { color:var(--cyan); }
.kpi.green .v { color:var(--lime); }
.alert { border-radius:12px; padding:12px 15px; margin-bottom:13px; font-weight:650;
         border:1px solid var(--line); background:var(--surface); color:var(--mute); }
.alert.on { color:#fff; background:linear-gradient(100deg,#ff66772b,#ff667708);
            border-color:#ff667766; animation:flash 1.6s ease-in-out 2; }
.alert.ok { color:var(--lime); border-color:#c8f16955; background:#c8f1690d; }
@keyframes flash { 50% { box-shadow:0 0 24px #ff667733; } }
.feed { display:flex; flex-direction:column; gap:8px; }
.card { display:flex; gap:11px; align-items:center; background:linear-gradient(110deg,#131e28,#101820);
        border:1px solid var(--line); border-left:3px solid var(--orange); border-radius:12px;
        padding:9px 10px; transition:transform .18s ease,border-color .18s ease; }
.card:hover { transform:translateX(2px); border-top-color:#3a4c55; border-right-color:#3a4c55; border-bottom-color:#3a4c55; }
.card.critical { border-left-color:var(--red); }
.card img { flex:none; width:84px; height:52px; object-fit:cover; border-radius:8px; background:#080d14; }
.card .t { font-weight:750; font-size:.88rem; } .card .m { color:var(--mute); font-size:.7rem; line-height:1.45; }
.badge { flex:none; margin-left:auto; font-size:.61rem; font-weight:850; padding:4px 7px;
         border-radius:6px; letter-spacing:.08em; }
.badge.critical { background:#ff66771c; color:#ff9da8; } .badge.high { background:#ffb4541c; color:#ffd08c; }
.empty { color:var(--mute); text-align:center; padding:26px 18px; border:1px dashed #34454e;
         border-radius:14px; background:#10192380; }
.stButton button { border-radius:10px; font-weight:750; transition:transform .16s ease,filter .16s ease; }
.stButton button:hover { transform:translateY(-1px); filter:brightness(1.12); }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--cyan),var(--lime)); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
@media (max-width:760px) {
    .block-container { padding-top:1.2rem; }
    .brand h1 { font-size:1.45rem; }
    .brand small { max-width:220px; font-size:.68rem; }
    .brand-mark { width:38px; height:38px; border-radius:12px; }
    .pill { padding:6px 9px; font-size:.61rem; }
    .kpis { gap:6px; }
    .kpi { padding:11px 9px; min-height:78px; }
    .kpi .l { font-size:.54rem; }
    .kpi .v { font-size:1.45rem; }
}
@media (prefers-reduced-motion:reduce) {
    *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


def new_client(api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url=DEPLOYMENT_URL, api_key=api_key)
    client.configure(InferenceConfiguration(api_key_transport="header"))
    return client


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Preparing baseline model weights at startup...")
def ensure_yolo_weights() -> Path:
    if YOLO_WEIGHTS_PATH.is_file() and file_sha256(YOLO_WEIGHTS_PATH) == YOLO_WEIGHTS_SHA256:
        return YOLO_WEIGHTS_PATH

    YOLO_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=YOLO_WEIGHTS_PATH.parent, suffix=".download"
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(YOLO_WEIGHTS_URL, timeout=60) as response:
                shutil.copyfileobj(response, temporary_file)

        if file_sha256(temporary_path) != YOLO_WEIGHTS_SHA256:
            raise RuntimeError("Downloaded YOLO26n weights failed SHA-256 verification.")
        os.replace(temporary_path, YOLO_WEIGHTS_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return YOLO_WEIGHTS_PATH


try:
    ensure_yolo_weights()
except (OSError, RuntimeError) as exc:
    st.warning(f"Could not prepare baseline model weights at startup: {exc}")


@st.cache_resource(show_spinner=" Preparing the Baseline model... ")
def load_yolo_model() -> Any:
    from ultralytics import YOLO

    return YOLO(str(ensure_yolo_weights()))


def infer(
    api_key: str,
    frame: np.ndarray,
    min_confidence: float,
    model_choice: str,
    yolo_model: Any | None,
) -> list[dict[str, Any]]:
    if model_choice == YOLO_BASELINE_MODEL:
        if yolo_model is None:
            raise RuntimeError("The YOLO26n baseline model has not been loaded.")
        with YOLO_INFERENCE_LOCK:
            result = yolo_model.predict(
                source=frame, conf=min_confidence, verbose=False
            )[0]
        if result.boxes is None:
            return []
        return [
            {
                "x": box[0],
                "y": box[1],
                "width": box[2],
                "height": box[3],
                "confidence": confidence,
                "class": str(result.names[int(class_id)]),
            }
            for box, confidence, class_id in zip(
                result.boxes.xywh.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            )
        ]

    result = new_client(api_key).run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": frame},
        use_cache=True,
    )
    block = result[0]["predictions"]
    rows = block["predictions"] if isinstance(block, dict) else block
    return [row for row in rows if row.get("confidence", 0) >= min_confidence]


def annotate(
    frame: np.ndarray, detections: list[dict[str, Any]], show_all_labels: bool = False
) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        name = det["class"]
        color = COLORS_BGR.get(name, (200, 200, 200))
        x1, y1 = int(det["x"] - det["width"] / 2), int(det["y"] - det["height"] / 2)
        x2, y2 = int(det["x"] + det["width"] / 2), int(det["y"] + det["height"] / 2)
        thickness = 3 if name in VIOLATIONS else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if name in VIOLATIONS or show_all_labels:
            label = f"{VIOLATIONS.get(name, name)} {det['confidence']:.0%}"
            cv2.putText(canvas, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas


def jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    return buffer.tobytes()


def clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def reset_state() -> None:
    st.session_state.incidents = []
    st.session_state.stats = {
        "helmet": 0, "without_helmet": 0, "vest": 0, "without_vest": 0,
        "people": 0, "objects": 0, "errors": 0,
    }
    st.session_state.last_alert = None
    st.session_state.status = "idle"


if "incidents" not in st.session_state:
    reset_state()


def kpi_html(model_choice: str) -> str:
    s = st.session_state.stats
    if model_choice == YOLO_BASELINE_MODEL:
        return (
            '<div class="kpis">'
            f'<div class="kpi cyan"><div class="l">Person boxes</div><div class="v">{s["people"]}</div></div>'
            f'<div class="kpi green"><div class="l">Object boxes</div><div class="v">{s["objects"]}</div></div>'
            "</div>"
        )

    inc = st.session_state.incidents
    helmets = s["helmet"] + s["without_helmet"]
    vests = s["vest"] + s["without_vest"]
    rate = 100 * (s["helmet"] + s["vest"]) / (helmets + vests) if helmets + vests else 100.0
    n_helmet = sum(i["Violation"] == "No Helmet" for i in inc)
    n_vest = sum(i["Violation"] == "No Safety Vest" for i in inc)
    return (
        '<div class="kpis">'
        f'<div class="kpi red"><div class="l">No Helmet alerts</div><div class="v">{n_helmet}</div></div>'
        f'<div class="kpi orange"><div class="l">No Vest alerts</div><div class="v">{n_vest}</div></div>'
        f'<div class="kpi green"><div class="l">PPE compliance</div><div class="v">{rate:.0f}%</div></div>'
        "</div>"
    )


def alert_html(model_choice: str) -> str:
    if model_choice == YOLO_BASELINE_MODEL:
        return '<div class="alert"> Baseline model selected </div>'
    last = st.session_state.last_alert
    if last:
        return (
            f'<div class="alert on">🚨 {last["Violation"]} detected · {last["Camera"]} · '
            f'{last["Video time"]}</div>'
        )
    if st.session_state.status == "live":
        return '<div class="alert ok">✔ All clear · monitoring in progress</div>'
    return '<div class="alert">Awaiting stream</div>'


def feed_html(model_choice: str, limit: int = 6) -> str:
    inc = st.session_state.incidents
    if not inc:
        if model_choice == YOLO_BASELINE_MODEL:
            return '<div class="empty">The baseline model is not trained to report PPE incidents.</div>'
        return '<div class="empty">No incidents recorded yet.</div>'
    cards = []
    for item in reversed(inc[-limit:]):
        sev = SEVERITY[item["Violation"]]
        cards.append(
            f'<div class="card {sev}"><img src="data:image/jpeg;base64,{item["_thumb"]}"/>'
            f'<div><div class="t">{item["Violation"]}</div>'
            f'<div class="m">{item["Camera"]} · video {item["Video time"]} · {item["Wall clock"]} · '
            f'{item["People"]} person(s) · {item["Confidence"]:.0%}</div></div>'
            f'<span class="badge {sev}">{sev.upper()}</span></div>'
        )
    return '<div class="feed">' + "".join(cards) + "</div>"


def pill_html() -> str:
    status = st.session_state.status
    label = {"live": "LIVE · ANALYSING", "done": "COMPLETED", "idle": "STANDBY"}[status]
    return f'<span class="pill {"live" if status == "live" else "done" if status == "done" else ""}">{label}</span>'


with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    model_choice = st.selectbox(
        "Inference model", [PPE_MODEL, YOLO_BASELINE_MODEL], index=0
    )
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    camera = st.text_input("Camera", value="CAM-01")
    upload = st.file_uploader("Upload CCTV recording", type=["mp4", "mov", "avi", "mkv"])
    speed = st.select_slider("Playback speed", options=[1, 2, 4, 8], value=1, format_func=lambda v: f"{v}×")
    sample_every = st.slider("Analyse a frame every (video s)", 0.5, 5.0, 1.0, 0.5)
    min_conf = st.slider("Min confidence", 0.1, 0.95, 0.5, 0.05)
    cooldown = st.slider("Alert cooldown (video s)", 2, 30, 8)
    if model_choice == PPE_MODEL and not api_key:
        st.caption("Service credentials must be configured before monitoring.")
    elif model_choice == YOLO_BASELINE_MODEL:
        st.caption("Pretrained; detects common objects, not helmets or safety vests.")
    start = st.button("▶ Start monitoring", type="primary", use_container_width=True,
                      disabled=upload is None or (model_choice == PPE_MODEL and not api_key))
    st.button("■ Stop", use_container_width=True)

header_slot = st.empty()


def render_header() -> None:
    model_label = "BASELINE" if model_choice == YOLO_BASELINE_MODEL else "TRAINED MODEL"
    header_slot.markdown(
        f'<div class="brand"><div class="brand-lockup"><span class="brand-mark">S</span><div>'
        f'<h1>Skipper</h1><small>FIELD SAFETY INTELLIGENCE · {camera} · {model_label}</small></div>'
        f'</div>{pill_html()}</div>',
        unsafe_allow_html=True,
    )


render_header()

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown('<div class="panel-title">Camera feed</div>', unsafe_allow_html=True)
    video_slot = st.empty()
    progress_slot = st.empty()
with right:
    alert_slot = st.empty()
    summary_title = (
        "Baseline detections"
        if model_choice == YOLO_BASELINE_MODEL
        else "Compliance snapshot"
    )
    st.markdown(f'<div class="panel-title">{summary_title}</div>', unsafe_allow_html=True)
    kpi_slot = st.empty()
    st.markdown('<div class="panel-title">Live incident feed</div>', unsafe_allow_html=True)
    feed_slot = st.empty()


def refresh_panels() -> None:
    alert_slot.markdown(alert_html(model_choice), unsafe_allow_html=True)
    kpi_slot.markdown(kpi_html(model_choice), unsafe_allow_html=True)
    feed_slot.markdown(feed_html(model_choice), unsafe_allow_html=True)


def handle_result(
    detections: list[dict[str, Any]], frame: np.ndarray, video_time: float,
    last_alert_at: dict[str, float], model_choice: str,
) -> None:
    stats = st.session_state.stats
    if model_choice == YOLO_BASELINE_MODEL:
        stats["objects"] += len(detections)
        stats["people"] += sum(detection["class"] == "person" for detection in detections)
        return

    counts: dict[str, int] = {}
    for det in detections:
        counts[det["class"]] = counts.get(det["class"], 0) + 1
    for name in ("helmet", "without_helmet", "vest", "without_vest"):
        stats[name] += counts.get(name, 0)

    for cls, label in VIOLATIONS.items():
        if cls not in counts:
            continue
        if video_time - last_alert_at.get(cls, -1e9) < cooldown:
            continue
        last_alert_at[cls] = video_time
        boxed = annotate(frame, detections)
        thumb = cv2.resize(boxed, (192, 108))
        incident = {
            "Camera": camera,
            "Video time": clock(video_time),
            "Wall clock": datetime.now().strftime("%H:%M:%S"),
            "Violation": label,
            "People": counts[cls],
            "Confidence": max(d["confidence"] for d in detections if d["class"] == cls),
            "_original": jpeg(frame),
            "_annotated": jpeg(boxed),
            "_thumb": base64.b64encode(jpeg(thumb, 70)).decode(),
        }
        st.session_state.incidents.append(incident)
        st.session_state.last_alert = incident
        st.toast(f"{label} · {camera} · {incident['Video time']}", icon="🚨")


def run_monitoring(video_bytes: bytes, suffix: str, model_choice: str) -> None:
    reset_state()
    yolo_model = None
    if model_choice == YOLO_BASELINE_MODEL:
        try:
            yolo_model = load_yolo_model()
        except (ImportError, OSError, RuntimeError) as exc:
            st.error(f"Could not prepare the Baseline model: {exc}")
            return

    st.session_state.status = "live"
    render_header()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(video_bytes)
        path = Path(handle.name)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        st.error("The uploaded video could not be decoded.")
        st.session_state.status = "idle"
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    pending: list[tuple[Future, np.ndarray, float]] = []
    latest: list[dict[str, Any]] = []
    latest_expiry = 0.0
    last_alert_at: dict[str, float] = {}
    next_sample = 0.0
    index = 0
    started = time.time()
    refresh_panels()

    def drain(block: bool = False) -> None:
        nonlocal latest, latest_expiry
        for item in list(pending):
            future, frame, video_time = item
            if not (future.done() or block):
                continue
            pending.remove(item)
            try:
                detections = future.result(timeout=60)
            except Exception:
                st.session_state.stats["errors"] += 1
                continue
            latest, latest_expiry = detections, video_time + 2 * sample_every
            handle_result(detections, frame, video_time, last_alert_at, model_choice)
            refresh_panels()

    max_in_flight = 1 if model_choice == YOLO_BASELINE_MODEL else MAX_IN_FLIGHT
    pool = ThreadPoolExecutor(max_workers=max_in_flight)
    try:
        while True:
            desired = int((time.time() - started) * speed * fps)
            while index < desired - 1 and capture.grab():
                index += 1
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            video_time = index / fps
            scale = INFERENCE_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (INFERENCE_WIDTH, round(frame.shape[0] * scale)))

            if video_time >= next_sample and len(pending) < max_in_flight:
                next_sample = video_time + sample_every
                pending.append(
                    (
                        pool.submit(
                            infer, api_key, frame.copy(), min_conf, model_choice, yolo_model
                        ),
                        frame.copy(),
                        video_time,
                    )
                )
            drain()

            view = (
                annotate(
                    frame, latest, show_all_labels=model_choice == YOLO_BASELINE_MODEL
                )
                if video_time <= latest_expiry else frame
            )
            cv2.rectangle(view, (0, 0), (360, 34), (0, 0, 0), -1)
            cv2.putText(view, f"{camera}  {clock(video_time)}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_slot.image(view, channels="BGR", output_format="JPEG", width="stretch")
            progress_slot.progress(min(1.0, index / total))
            time.sleep(max(0.0, (index / fps / speed) - (time.time() - started)))
        drain(block=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        capture.release()
        path.unlink(missing_ok=True)
    st.session_state.status = "done"
    st.session_state.last_alert = None
    render_header()
    refresh_panels()
    progress_slot.empty()


if start and upload is not None:
    run_monitoring(upload.getvalue(), Path(upload.name).suffix or ".mp4", model_choice)
else:
    refresh_panels()
    if st.session_state.status == "idle":
        video_slot.markdown(
            '<div class="empty" style="padding:120px 20px">Upload a CCTV recording and press '
            "<b>Start monitoring</b> to begin the live compliance analysis.</div>",
            unsafe_allow_html=True,
        )

if st.session_state.stats["errors"]:
    st.warning(
        f"{st.session_state.stats['errors']} frame(s) failed during analysis. "
        + (
            "Check the local model and device configuration."
            if model_choice == YOLO_BASELINE_MODEL
            else "Check the application configuration and connectivity."
        )
    )

incidents = st.session_state.incidents
if incidents and st.session_state.status != "live":
    st.divider()
    st.markdown("### Incident log")
    table = pd.DataFrame([{k: v for k, v in i.items() if not k.startswith("_")} for i in incidents])
    table["Confidence"] = table["Confidence"].map(lambda v: f"{v:.0%}")
    selection = st.dataframe(
        table, hide_index=True, width="stretch", height=260,
        on_select="rerun", selection_mode="single-row", key="incident_table",
    )
    rows = selection.selection.rows
    chosen = incidents[rows[0]] if rows else incidents[0]
    st.caption("Select a row to review its screenshots." if rows else "Showing the first incident. Select a row to switch.")
    c1, c2 = st.columns(2)
    c1.image(chosen["_original"], caption="Original frame", width="stretch")
    c2.image(chosen["_annotated"], caption="Detections", width="stretch")
    st.download_button("⬇ Export incidents (CSV)", table.to_csv(index=False), "incidents.csv", "text/csv")
