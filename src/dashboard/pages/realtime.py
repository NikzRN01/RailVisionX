"""Real-time deblurring dashboard page — Enhanced UI with Plotly live charts."""

import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.enhancement.deblur_net_torch import DeblurMobileNetV2UNet
from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
from src.quality.lowlight_score import mean_intensity
from src.dashboard.theme import (
    inject_theme, section_header,
    ACCENT_CYAN, ACCENT_AMBER, SUCCESS, DANGER,
    TEXT_MUTED, TEXT_PRIMARY, BG_CARD, BG_DEEP, BORDER,
)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_INPUT_SIZE = (256, 256)
DEFAULT_MODEL_PATHS = [
    Path("models/checkpoints/deblur_mobilenetv2_unet_torch_best.pt"),
    Path("outputs/result/best_model.pt"),
]

# ── Plotly dark layout defaults ───────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(device: str):
    """Load and cache the deblur model.

    Searches for trained weights in default locations. Falls back to ImageNet-pretrained
    encoder if no trained weights found.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in eval mode
    """
    model = DeblurMobileNetV2UNet(weights=None, backbone_trainable=False).to(device)
    weights_found = next((p for p in DEFAULT_MODEL_PATHS if p.exists()), None)
    if weights_found:
        try:
            state = torch.load(weights_found, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            st.sidebar.success(f"✅ Model: {weights_found.name}")
        except Exception as e:
            st.sidebar.error(f"❌ Model could not load: {str(e)[:50]}")
            st.sidebar.warning("The live camera can still run, but its output is not reliable deblurring until the model is repaired.")
    else:
        st.sidebar.warning("No compatible trained deblurring model is linked yet.")
        st.sidebar.info("The live camera and video workflow will run, but output uses a fallback encoder rather than your trained deblurring model.")
    model.eval()
    return model


# ── Frame processing ──────────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess frame for model inference.

    Args:
        frame: Input frame (H, W, 3) BGR from OpenCV

    Returns:
        Preprocessed tensor (1, 3, H, W)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, MODEL_INPUT_SIZE)
    t = torch.from_numpy(frame_resized).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def postprocess_frame(output_tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
    """Postprocess model output to displayable frame.

    Args:
        output_tensor: Model output tensor (1, 3, H, W) in [0, 1]
        original_size: Original frame size (H, W)

    Returns:
        RGB frame (H, W, 3) as uint8
    """
    out = output_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    return cv2.resize(out, (original_size[1], original_size[0]))


def deblur_frame(model, frame: np.ndarray, device: str):
    """Apply deblurring to a frame.

    Args:
        model: Deblur model
        frame: Input frame BGR from OpenCV
        device: Device to run inference on

    Returns:
        Tuple of (deblurred frame RGB, inference time in seconds)
    """
    t0 = time.time()
    inp = preprocess_frame(frame).to(device)
    with torch.no_grad():
        out = model(inp)
    deblurred = postprocess_frame(out, frame.shape[:2])
    return deblurred, time.time() - t0


def calculate_frame_metrics(blurred_frame: np.ndarray, deblurred_frame: np.ndarray,
                            blurred_is_bgr: bool = True) -> dict:
    """Calculate comprehensive quality metrics comparing blurred and deblurred frames.

    Args:
        blurred_frame: Input blurred frame (RGB or BGR).
        deblurred_frame: Deblurred output frame (always RGB from postprocess_frame).
        blurred_is_bgr: Set True when blurred_frame comes directly from cv2.VideoCapture
                        (BGR byte order). Set False when it is already RGB (e.g. a saved
                        snapshot retrieved from session_state).

    Returns:
        Dictionary with all metrics and improvements, keyed by:
        'blurred', 'deblurred', 'comparison', 'improvements'
    """
    # Convert to RGB only when the caller confirms the frame is BGR.
    if blurred_is_bgr and blurred_frame.ndim == 3 and blurred_frame.shape[2] == 3:
        blurred_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
    else:
        blurred_rgb = blurred_frame

    if blurred_rgb.shape != deblurred_frame.shape:
        blurred_rgb = cv2.resize(blurred_rgb, (deblurred_frame.shape[1], deblurred_frame.shape[0]))

    b_norm = blurred_rgb.astype(np.float32) / 255.0
    d_norm = deblurred_frame.astype(np.float32) / 255.0

    b_gray = cv2.cvtColor((b_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    d_gray = cv2.cvtColor((d_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

    b_lap  = variance_of_laplacian(b_norm)
    d_lap  = variance_of_laplacian(d_norm)
    b_grad = mean_gradient_magnitude(b_norm)
    d_grad = mean_gradient_magnitude(d_norm)

    ssim_score = ssim(b_gray, d_gray, data_range=1.0)
    psnr_score = psnr(b_norm, d_norm, data_range=1.0)
    mae_score  = float(np.mean(np.abs(b_norm - d_norm)))
    px_acc     = float(np.mean(np.abs(b_norm - d_norm) < 0.1) * 100)

    b_bright = mean_intensity(b_gray)
    d_bright = mean_intensity(d_gray)

    def imp(n, o): return ((n - o) / o * 100) if o else 0.0

    return {
        "blurred":    {"laplacian_variance": b_lap, "gradient_magnitude": b_grad, "brightness": b_bright},
        "deblurred":  {"laplacian_variance": d_lap, "gradient_magnitude": d_grad, "brightness": d_bright},
        "comparison": {"ssim": ssim_score, "psnr": psnr_score, "mae": mae_score, "pixel_accuracy": px_acc},
        "improvements": {"sharpness": imp(d_lap, b_lap), "gradient": imp(d_grad, b_grad),
                         "brightness": imp(d_bright, b_bright)},
    }


# ── Plotly chart builders ─────────────────────────────────────────────────────
def build_fps_gauge(fps: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fps,
        number={"suffix": " fps", "font": {"color": ACCENT_CYAN, "size": 26, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 60], "tickcolor": TEXT_MUTED, "tickfont": {"size": 10}},
            "bar": {"color": ACCENT_CYAN},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 15],  "color": DANGER + "33"},
                {"range": [15, 30], "color": ACCENT_AMBER + "33"},
                {"range": [30, 60], "color": SUCCESS + "33"},
            ],
            "threshold": {"line": {"color": ACCENT_AMBER, "width": 2}, "value": 30},
        },
        title={"text": "FPS", "font": {"color": TEXT_MUTED, "size": 12}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT_PRIMARY,
                      margin=dict(l=20, r=20, t=40, b=20), height=200)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert a hex color string like '#00d4ff' to 'rgba(0,212,255,0.08)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_live_chart(frames_data: list) -> go.Figure:
    if not frames_data:
        return go.Figure()
    xs      = [f["frame_num"] for f in frames_data]
    psnrs   = [f["metrics"]["comparison"]["psnr"] for f in frames_data]
    ssims   = [f["metrics"]["comparison"]["ssim"] for f in frames_data]
    sharpB  = [f["metrics"]["blurred"]["laplacian_variance"] for f in frames_data]
    sharpD  = [f["metrics"]["deblurred"]["laplacian_variance"] for f in frames_data]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("PSNR (dB)", "SSIM", "Sharpness — Laplacian", "Inference Time (ms)"),
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    def line(x, y, name, color, row, col, fill=False):
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy" if fill else "none",
            fillcolor=_hex_to_rgba(color, 0.08) if fill else None,
        ), row=row, col=col)

    line(xs, psnrs,  "PSNR",          ACCENT_CYAN,  1, 1, fill=True)
    line(xs, ssims,  "SSIM",          ACCENT_AMBER, 1, 2, fill=True)
    line(xs, sharpB, "Blurred",       DANGER,       2, 1)
    line(xs, sharpD, "Deblurred",     SUCCESS,      2, 1)
    infer_ms = [f["inference_time"] for f in frames_data]
    line(xs, infer_ms, "Infer ms",    TEXT_MUTED,   2, 2, fill=True)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_MUTED
        ann.font.size  = 11
    return fig


def build_improvement_bar(metrics: dict) -> go.Figure:
    imps = metrics["improvements"]
    labels = ["Sharpness", "Gradient", "Brightness"]
    values = [imps["sharpness"], imps["gradient"], imps["brightness"]]
    colors = [SUCCESS if v >= 0 else DANGER for v in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=200,
        yaxis_title="Improvement %",
        yaxis_ticksuffix="%",
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED, line_width=1)
    return fig


# ── Inline metric row HTML ────────────────────────────────────────────────────
def metric_row_html(metrics: dict, fps: float, frame_num: int, inf_ms: float) -> str:
    c = metrics["comparison"]
    imp = metrics["improvements"]

    rows = [
        ("PSNR",        f"{c['psnr']:.2f} dB",           "higher ↑ is better"),
        ("SSIM",        f"{c['ssim']:.4f}",               "0–1, higher is better"),
        ("MAE",         f"{c['mae']:.4f}",                "lower ↓ is better"),
        ("Px Acc",      f"{c['pixel_accuracy']:.1f}%",    "within 10% threshold"),
        ("Sharpness ↑", f"{imp['sharpness']:+.1f}%",      "deblurred vs blurred"),
    ]
    trs = "".join(f"""
    <tr>
        <td style='padding:5px 12px;color:{TEXT_MUTED};font-size:0.82rem;'>{k}</td>
        <td style='padding:5px 12px;color:{ACCENT_CYAN};font-family:JetBrains Mono,monospace;
                   font-size:0.88rem;font-weight:600;'>{v}</td>
        <td style='padding:5px 12px;color:{TEXT_MUTED};font-size:0.75rem;'>{note}</td>
    </tr>""" for k, v, note in rows)

    return f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;
                padding:12px 16px;margin-bottom:8px;">
        <div style="display:flex;gap:24px;margin-bottom:10px;flex-wrap:wrap;">
            <span style="color:{TEXT_MUTED};font-size:0.78rem;">Frame
                <span style="color:{TEXT_PRIMARY};font-weight:600;">{frame_num}</span></span>
            <span style="color:{TEXT_MUTED};font-size:0.78rem;">FPS
                <span style="color:{ACCENT_CYAN};font-weight:700;">{fps:.1f}</span></span>
            <span style="color:{TEXT_MUTED};font-size:0.78rem;">Inference
                <span style="color:{ACCENT_AMBER};font-weight:700;">{inf_ms:.1f}ms</span></span>
        </div>
        <table style="border-collapse:collapse;width:100%;">{trs}</table>
    </div>"""


# ── Main page ─────────────────────────────────────────────────────────────────
def main():
    inject_theme()
    st.markdown(f"""
    <h1 style="margin-bottom:4px;">🎬 Enhance a video</h1>
    <p style="color:{TEXT_MUTED};margin-bottom:24px;">
        Choose a webcam or upload a clip. Live quality metrics are available while it runs.
    </p>
    """, unsafe_allow_html=True)

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
    if "saved_frames" not in st.session_state:
        st.session_state.saved_frames = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Device pill
    d_color = SUCCESS if device == "cuda" else ACCENT_AMBER
    from src.dashboard.theme import status_pill
    st.markdown(status_pill(f"Device: {device.upper()}", d_color), unsafe_allow_html=True)

    model = load_model(device)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.markdown(f"<h4 style='color:{ACCENT_CYAN};'>📹 Input Source</h4>",
                        unsafe_allow_html=True)
    input_source = st.sidebar.radio("Source", ["Use webcam", "Upload a video"],
                                    label_visibility="collapsed")

    frames_data = []

    # ══════════════════════════════════════════════════════════════════════════
    # WEBCAM
    # ══════════════════════════════════════════════════════════════════════════
    if input_source == "Use webcam":
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;
                    padding:22px;margin:10px 0 20px;max-width:760px;">
            <div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;margin-bottom:6px;">Use your webcam for live enhancement</div>
            <div style="color:{TEXT_MUTED};font-size:0.9rem;line-height:1.5;">
                Click Start Live Camera, allow browser camera access, and RailVisionX will show each input frame beside its enhanced result.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Start Live Camera", type="primary", width="content"):
            st.session_state.webcam_running = True
            st.session_state.capture_start_time = time.time()
            st.rerun()

        st.sidebar.markdown(f"<h4 style='color:{ACCENT_CYAN};'>⏱ Capture Duration</h4>",
                            unsafe_allow_html=True)
        use_duration     = st.sidebar.checkbox("Auto-stop after duration", value=False)
        duration_seconds = st.sidebar.slider("Duration (s)", 5, 300, 30, 5,
                                             disabled=not use_duration)

        col_start, col_stop = st.sidebar.columns(2)
        if col_start.button("▶ Start", disabled=st.session_state.webcam_running):
            st.session_state.webcam_running   = True
            st.session_state.capture_start_time = time.time()
            st.rerun()
        if col_stop.button("⏹ Stop", disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False
            st.rerun()

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam.")
                st.session_state.webcam_running = False
                return

            snapshot_btn = st.sidebar.button("📸 Save Snapshot")

            # Layout
            img_col, metric_col = st.columns([3, 2])
            with img_col:
                blurred_ph  = st.empty()
                sharp_ph    = st.empty()
            with metric_col:
                gauge_ph    = st.empty()
                impr_ph     = st.empty()
                metric_ph   = st.empty()

            chart_ph    = st.empty()
            dur_ph      = st.empty()
            stop_btn    = st.button("🛑 Stop Capture")
            if stop_btn:
                st.session_state.webcam_running = False

            frame_count = 0
            fps_vals    = []

            try:
                while st.session_state.webcam_running:
                    if use_duration:
                        elapsed   = time.time() - st.session_state.capture_start_time
                        remaining = duration_seconds - elapsed
                        if elapsed >= duration_seconds:
                            st.session_state.webcam_running = False
                            dur_ph.success(f"✅ Auto-stopped after {duration_seconds}s")
                            break
                        pct = int(elapsed / duration_seconds * 100)
                        dur_ph.progress(pct, text=f"⏱ {elapsed:.0f}s / {duration_seconds}s")

                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠ Frame capture failed.")
                        break

                    deblurred, inf_t = deblur_frame(model, frame, device)
                    fps = 1 / inf_t if inf_t > 0 else 0
                    fps_vals.append(fps)

                    metrics = calculate_frame_metrics(frame, deblurred)
                    frames_data.append({
                        "frame_num":      frame_count,
                        "inference_time": inf_t * 1000,
                        "metrics":        metrics,
                    })

                    # Images — side by side inside img_col
                    with img_col:
                        r1, r2 = st.columns(2)
                        r1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 caption="Blurred Input", width="stretch")
                        r2.image(deblurred, caption="Deblurred Output",
                                 width="stretch")

                    # Right column: gauge + improvement bar + metric table
                    with metric_col:
                        gauge_ph.plotly_chart(build_fps_gauge(fps),
                                              width="stretch", key=f"gauge_{frame_count}")
                        impr_ph.plotly_chart(build_improvement_bar(metrics),
                                             width="stretch", key=f"impr_{frame_count}")
                        metric_ph.markdown(
                            metric_row_html(metrics, fps, frame_count, inf_t * 1000),
                            unsafe_allow_html=True,
                        )

                    # Live chart (last 50 frames)
                    if frame_count % 5 == 0 and len(frames_data) > 1:
                        chart_ph.plotly_chart(
                            build_live_chart(frames_data[-50:]),
                            width="stretch", key=f"chart_{frame_count}",
                        )

                    if snapshot_btn:
                        st.session_state.saved_frames.append({
                            "frame_num": frame_count,
                            "blurred":   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy(),
                            "deblurred": deblurred.copy(),
                            "metrics":   metrics,
                        })
                        st.sidebar.success(f"📸 Saved frame {frame_count}")

                    frames_data = frames_data[-100:]
                    fps_vals    = fps_vals[-100:]
                    frame_count += 1

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                cap.release()
                st.session_state.webcam_running = False

            if frame_count > 0:
                st.success(f"✅ Processed {frame_count} frames · Avg FPS: {np.mean(fps_vals):.1f}")

            _csv_export(frames_data)

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown(f"<h3 style='color:{TEXT_PRIMARY};'>📁 Upload Video</h3>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

        if uploaded:
            if st.button("▶ Process Video"):
                suffix = Path(uploaded.name).suffix.lower() or ".mp4"
                tmp = None
                cap = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded.getvalue())
                        tmp = Path(tmp_file.name)

                    cap = cv2.VideoCapture(str(tmp))
                    if not cap.isOpened():
                        st.error("❌ The uploaded video could not be opened. Try a supported MP4, AVI, MOV, or MKV file.")
                        return

                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.info(f"🎞 Loaded **{total}** frames")
                    prog = st.progress(0, text="Processing…")
                    c_left, c_right = st.columns(2)
                    frame_count = 0
                    fps_vals = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        deblurred, inf_t = deblur_frame(model, frame, device)
                        fps = 1 / inf_t if inf_t > 0 else 0
                        fps_vals.append(fps)
                        metrics = calculate_frame_metrics(frame, deblurred, blurred_is_bgr=True)
                        frames_data.append({
                            "frame_num":      frame_count,
                            "inference_time": inf_t * 1000,
                            "metrics":        metrics,
                        })

                        if frame_count % 10 == 0:
                            with c_left:
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                         caption=f"Blurred · Frame {frame_count}/{total}",
                                         width="stretch")
                            with c_right:
                                st.image(deblurred,
                                         caption=f"Deblurred · Frame {frame_count}/{total}",
                                         width="stretch")
                            prog.progress(frame_count / max(total, 1),
                                          text=f"Frame {frame_count}/{total} · {fps:.1f} fps")

                        frame_count += 1

                finally:
                    if cap is not None:
                        cap.release()
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)

                if not frames_data:
                    st.error("❌ No video frames could be decoded from the uploaded file.")
                    return

                avg_fps    = float(np.mean(fps_vals))
                total_time = sum(f["inference_time"] for f in frames_data) / 1000
                st.success(
                    f"✅ Done · {frame_count} frames · Avg FPS: {avg_fps:.1f} · Time: {total_time:.1f}s"
                )

                # Post-processing charts
                section_header("📊 Performance Analysis", "Full-video metrics breakdown")
                st.plotly_chart(build_live_chart(frames_data),
                                width="stretch")
                _csv_export(frames_data)

    # ── Saved snapshots ───────────────────────────────────────────────────────
    _show_snapshots()


def _csv_export(frames_data: list):
    if not frames_data:
        return
    if st.button("💾 Export Metrics to CSV"):
        import pandas as pd
        rows = [{
            "frame":                   f["frame_num"],
            "inference_ms":            f["inference_time"],
            "blurred_laplacian":       f["metrics"]["blurred"]["laplacian_variance"],
            "blurred_gradient":        f["metrics"]["blurred"]["gradient_magnitude"],
            "blurred_brightness":      f["metrics"]["blurred"]["brightness"],
            "deblurred_laplacian":     f["metrics"]["deblurred"]["laplacian_variance"],
            "deblurred_gradient":      f["metrics"]["deblurred"]["gradient_magnitude"],
            "deblurred_brightness":    f["metrics"]["deblurred"]["brightness"],
            "ssim":                    f["metrics"]["comparison"]["ssim"],
            "psnr":                    f["metrics"]["comparison"]["psnr"],
            "mae":                     f["metrics"]["comparison"]["mae"],
            "pixel_accuracy":          f["metrics"]["comparison"]["pixel_accuracy"],
            "sharpness_improvement_%": f["metrics"]["improvements"]["sharpness"],
            "gradient_improvement_%":  f["metrics"]["improvements"]["gradient"],
            "brightness_improvement_%":f["metrics"]["improvements"]["brightness"],
        } for f in frames_data]
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "realtime_metrics.csv", "text/csv")


def _show_snapshots():
    if not st.session_state.saved_frames:
        return
    st.markdown("<hr>", unsafe_allow_html=True)
    from src.dashboard.theme import TEXT_PRIMARY
    st.markdown(
        f"<h3 style='color:{TEXT_PRIMARY};'>📸 Saved Snapshots "
        f"({len(st.session_state.saved_frames)})</h3>",
        unsafe_allow_html=True,
    )
    for idx, snap in enumerate(st.session_state.saved_frames):
        with st.expander(f"Snapshot {idx+1} · Frame {snap['frame_num']}", expanded=False):
            c1, c2 = st.columns(2)
            c1.image(snap["blurred"], caption="Blurred", width="stretch")
            c2.image(snap["deblurred"], caption="Deblurred", width="stretch")
            m = snap["metrics"]
            st.plotly_chart(build_improvement_bar(m), width="stretch",
                            key=f"snap_bar_{idx}")
            st.markdown(
                metric_row_html(m, 0, snap["frame_num"], 0),
                unsafe_allow_html=True,
            )
    if st.button("🗑 Clear Snapshots"):
        st.session_state.saved_frames = []
        st.rerun()


if __name__ == "__main__":
    main()
