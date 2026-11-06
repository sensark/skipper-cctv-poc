from __future__ import annotations

import base64
import os
import tempfile
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

warnings.filterwarnings("ignore", message=".*legacy channel.*")

DEPLOYMENT_URL = "https://skmodel.roboflow.cloud"
WORKSPACE = "sensark"
WORKFLOW = (
    "construction-site-po4lz-1-rfdetr-small-t1-"
    "vconstruction-site-po4lz-1-rfdetr-small-t1-logic"
)
INFERENCE_WIDTH = 1280
MAX_IN_FLIGHT = 4

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

st.set_page_config(page_title="SiteGuard AI | PPE Compliance", page_icon="🦺", layout="wide")

st.markdown(
    """
<style>
:root { --bg:#0b1220; --card:#121a2c; --line:#1f2a44; --txt:#e6ebf5; --mute:#8a97b5;
        --red:#ef4444; --orange:#f97316; --green:#10b981; --blue:#38bdf8; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #14213d 0%, var(--bg) 55%); color:var(--txt); }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background:#0d1527; border-right:1px solid var(--line); }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
.brand { display:flex; align-items:center; justify-content:space-between; margin-bottom:.8rem; }
.brand h1 { font-size:1.7rem; margin:0; font-weight:800; letter-spacing:.3px;
            background:linear-gradient(90deg,#38bdf8,#818cf8 60%,#f472b6); -webkit-background-clip:text; color:transparent; }
.brand small { color:var(--mute); display:block; margin-top:2px; }
.pill { padding:6px 14px; border-radius:999px; font-weight:700; font-size:.78rem; letter-spacing:.6px;
        border:1px solid var(--line); background:var(--card); color:var(--mute); }
.pill.live { color:#fff; background:rgba(239,68,68,.15); border-color:rgba(239,68,68,.6); }
.pill.live::before { content:""; display:inline-block; width:9px; height:9px; margin-right:8px; border-radius:50%;
        background:var(--red); animation:pulse 1.2s infinite; }
.pill.done { color:var(--green); border-color:rgba(16,185,129,.5); background:rgba(16,185,129,.1); }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(239,68,68,.7)} 70%{box-shadow:0 0 0 10px rgba(239,68,68,0)} 100%{box-shadow:0 0 0 0 rgba(239,68,68,0)} }
.kpis { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.kpi { background:linear-gradient(145deg,#141f36,#0f1729); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.kpi .l { color:var(--mute); font-size:.72rem; text-transform:uppercase; letter-spacing:1px; }
.kpi .v { font-size:1.9rem; font-weight:800; line-height:1.25; }
.kpi.red .v { color:var(--red); } .kpi.orange .v { color:var(--orange); }
.kpi.green .v { color:var(--green); } .kpi.blue .v { color:var(--blue); }
.alert { border-radius:12px; padding:12px 16px; margin-bottom:12px; font-weight:600; border:1px solid var(--line); background:var(--card); color:var(--mute); }
.alert.on { color:#fff; background:linear-gradient(90deg,rgba(239,68,68,.35),rgba(239,68,68,.08)); border-color:rgba(239,68,68,.7); animation:flash 1.4s ease-in-out 3; }
.alert.ok { color:var(--green); border-color:rgba(16,185,129,.4); background:rgba(16,185,129,.08); }
@keyframes flash { 50% { box-shadow:0 0 22px rgba(239,68,68,.55); } }
.panel-title { color:var(--mute); text-transform:uppercase; letter-spacing:1.2px; font-size:.75rem; margin:6px 0 8px; }
.feed { display:flex; flex-direction:column; gap:8px; }
.card { display:flex; gap:12px; align-items:center; background:var(--card); border:1px solid var(--line);
        border-left:4px solid var(--orange); border-radius:12px; padding:8px 12px; }
.card.critical { border-left-color:var(--red); }
.card img { width:96px; height:54px; object-fit:cover; border-radius:8px; }
.card .t { font-weight:700; } .card .m { color:var(--mute); font-size:.78rem; }
.badge { margin-left:auto; font-size:.68rem; font-weight:800; padding:3px 9px; border-radius:6px; letter-spacing:.8px; }
.badge.critical { background:rgba(239,68,68,.18); color:#fca5a5; } .badge.high { background:rgba(249,115,22,.18); color:#fdba74; }
.empty { color:var(--mute); text-align:center; padding:24px; border:1px dashed var(--line); border-radius:12px; }
</style>
""",
    unsafe_allow_html=True,
)


def new_client(api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url=DEPLOYMENT_URL, api_key=api_key)
    client.configure(InferenceConfiguration(api_key_transport="header"))
    return client


def infer(api_key: str, frame: np.ndarray, min_confidence: float) -> list[dict[str, Any]]:
    result = new_client(api_key).run_workflow(
        workspace_name=WORKSPACE, workflow_id=WORKFLOW, images={"image": frame}
    )
    block = result[0]["predictions"]
    rows = block["predictions"] if isinstance(block, dict) else block
    return [row for row in rows if row.get("confidence", 0) >= min_confidence]


def annotate(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        name = det["class"]
        color = COLORS_BGR.get(name, (200, 200, 200))
        x1, y1 = int(det["x"] - det["width"] / 2), int(det["y"] - det["height"] / 2)
        x2, y2 = int(det["x"] + det["width"] / 2), int(det["y"] + det["height"] / 2)
        thickness = 3 if name in VIOLATIONS else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if name in VIOLATIONS:
            label = f"{VIOLATIONS[name]} {det['confidence']:.0%}"
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
        "frames": 0, "helmet": 0, "without_helmet": 0, "vest": 0, "without_vest": 0, "errors": 0,
    }
    st.session_state.last_alert = None
    st.session_state.status = "idle"


if "incidents" not in st.session_state:
    reset_state()


def kpi_html() -> str:
    s = st.session_state.stats
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
        f'<div class="kpi blue"><div class="l">Frames analysed</div><div class="v">{s["frames"]}</div></div>'
        "</div>"
    )


def alert_html() -> str:
    last = st.session_state.last_alert
    if last:
        return (
            f'<div class="alert on">🚨 {last["Violation"]} detected · {last["Camera"]} · '
            f'{last["Video time"]}</div>'
        )
    if st.session_state.status == "live":
        return '<div class="alert ok">✔ All clear · monitoring in progress</div>'
    return '<div class="alert">Awaiting stream</div>'


def feed_html(limit: int = 6) -> str:
    inc = st.session_state.incidents
    if not inc:
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
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    camera = st.text_input("Camera", value="CAM-01")
    upload = st.file_uploader("Upload CCTV recording", type=["mp4", "mov", "avi", "mkv"])
    speed = st.select_slider("Playback speed", options=[1, 2, 4, 8], value=1, format_func=lambda v: f"{v}×")
    sample_every = st.slider("Analyse a frame every (video s)", 0.5, 5.0, 1.0, 0.5)
    min_conf = st.slider("Min confidence", 0.1, 0.95, 0.5, 0.05)
    cooldown = st.slider("Alert cooldown (video s)", 2, 30, 8)
    if not api_key:
        st.caption("Service credentials must be configured before monitoring.")
    start = st.button("▶ Start monitoring", type="primary", use_container_width=True,
                      disabled=upload is None or not api_key)
    st.button("■ Stop", use_container_width=True)

header_slot = st.empty()


def render_header() -> None:
    header_slot.markdown(
        f'<div class="brand"><div><h1>🦺 SiteGuard AI</h1><small>Real-time PPE compliance monitoring · '
        f'{camera}</small></div>{pill_html()}</div>',
        unsafe_allow_html=True,
    )


render_header()

left, right = st.columns([3, 2], gap="large")
with left:
    video_slot = st.empty()
    progress_slot = st.empty()
with right:
    alert_slot = st.empty()
    kpi_slot = st.empty()
    st.markdown('<div class="panel-title">Live incident feed</div>', unsafe_allow_html=True)
    feed_slot = st.empty()


def refresh_panels() -> None:
    alert_slot.markdown(alert_html(), unsafe_allow_html=True)
    kpi_slot.markdown(kpi_html(), unsafe_allow_html=True)
    feed_slot.markdown(feed_html(), unsafe_allow_html=True)


def handle_result(
    detections: list[dict[str, Any]], frame: np.ndarray, video_time: float,
    last_alert_at: dict[str, float],
) -> None:
    stats = st.session_state.stats
    stats["frames"] += 1
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


def run_monitoring(video_bytes: bytes, suffix: str) -> None:
    reset_state()
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
            handle_result(detections, frame, video_time, last_alert_at)
            refresh_panels()

    pool = ThreadPoolExecutor(max_workers=MAX_IN_FLIGHT)
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

            if video_time >= next_sample and len(pending) < MAX_IN_FLIGHT:
                next_sample = video_time + sample_every
                pending.append((pool.submit(infer, api_key, frame.copy(), min_conf), frame.copy(), video_time))
            drain()

            view = annotate(frame, latest) if video_time <= latest_expiry else frame
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
    run_monitoring(upload.getvalue(), Path(upload.name).suffix or ".mp4")
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
        "Check the application configuration and connectivity."
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
