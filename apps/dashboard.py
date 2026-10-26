"""
RailVisionX Dashboard - Main Entry Point (Enhanced UI)
Multi-page Streamlit application for real-time deblurring analysis.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard.theme import inject_theme, status_pill, BG_CARD, ACCENT_CYAN, TEXT_MUTED, SUCCESS, ACCENT_AMBER, BORDER, TEXT_PRIMARY


def _navigate(destination: str) -> None:
    """Switch the sidebar task before Streamlit renders the next run."""
    st.session_state.dashboard_navigation = destination


def main():
    st.set_page_config(
        page_title="RailVisionX · Vision Dashboard",
        page_icon="🚆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # ── Sidebar navigation ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:12px 0 20px;">
            <div style="font-size:2.2rem;">🚆</div>
            <div style="color:{ACCENT_CYAN};font-size:1.2rem;font-weight:700;
                        letter-spacing:0.05em;">RailVisionX</div>
            <div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.12em;margin-top:2px;">Vision Dashboard</div>
        </div>
        <hr style="border-color:{BORDER};margin:0 0 16px;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Choose a task",
            ["🏠  Home", "🎬  Enhance a video", "🔍  Compare two images", "📊  Explore image quality"],
            label_visibility="collapsed",
            key="dashboard_navigation",
        )

        st.markdown(f"""
        <hr style="border-color:{BORDER};margin:20px 0 12px;">
        <div style="color:{TEXT_MUTED};font-size:0.7rem;text-align:center;
                    text-transform:uppercase;letter-spacing:0.1em;">
            Start simple. Detailed metrics are available inside each tool.
        </div>
        """, unsafe_allow_html=True)

    # ── Page routing ──────────────────────────────────────────────────────────
    if "Home" in page:
        show_home()
    elif "Enhance" in page:
        from src.dashboard.pages.realtime import main as realtime_main
        realtime_main()
    elif "Compare" in page:
        from src.dashboard.pages.compare_frames import main as compare_main
        compare_main()
    elif "Explore" in page:
        from src.dashboard.pages.top10 import main as topk_main
        topk_main()


def show_home():
    """Task-first landing page for people opening the dashboard for the first time."""
    import torch

    def image_count(folder: Path) -> int:
        if not folder.exists():
            return 0
        return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    test_pairs = min(image_count(Path("data/split/test/blurred")),
                     image_count(Path("data/split/test/sharp")))
    raw_pairs = min(image_count(Path("data/raw/blurred")),
                    image_count(Path("data/raw/sharp")))
    standard_model = any(path.exists() for path in (
        Path("models/checkpoints/deblur_mobilenetv2_unet_torch_best.pt"),
        Path("outputs/result/best_model.pt"),
    ))

    st.markdown(f"""
    <div style="margin:8px 0 26px;max-width:760px;">
        <div style="color:{ACCENT_CYAN};font-size:0.78rem;font-weight:700;letter-spacing:0.12em;">RAILVISIONX</div>
        <h1 style="margin:7px 0 10px;font-size:2.3rem;">Make blurry rail footage easier to inspect.</h1>
        <p style="color:{TEXT_MUTED};font-size:1rem;line-height:1.6;margin:0;">
            Start with one clear task. Advanced charts and metrics are available only when you need to dig deeper.
        </p>
    </div>
    """, unsafe_allow_html=True)

    device_label = "GPU ready" if torch.cuda.is_available() else "CPU mode"
    device_color = SUCCESS if torch.cuda.is_available() else ACCENT_AMBER
    model_label = "Model file found" if standard_model else "Model needs setup"
    model_color = SUCCESS if standard_model else ACCENT_AMBER
    st.markdown(
        status_pill(device_label, device_color)
        + status_pill(f"{test_pairs} test image pairs", ACCENT_CYAN)
        + status_pill(model_label, model_color),
        unsafe_allow_html=True,
    )

    st.markdown("### What would you like to do?")
    col1, col2, col3 = st.columns(3)
    choices = [
        (col1, "🎬", "Enhance a video", "Use your webcam or upload a video, then see the original and enhanced frames.", "Open video enhancement", "🎬  Enhance a video"),
        (col2, "🔍", "Compare two images", "Put a blurred image next to a restored or reference image and inspect the difference.", "Open image comparison", "🔍  Compare two images"),
        (col3, "📊", "Explore image quality", "Find the clearest or blurriest images in the prepared test set.", "Open quality explorer", "📊  Explore image quality"),
    ]
    for column, icon, title, description, label, destination in choices:
        with column:
            st.markdown(f"""
            <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;padding:22px;min-height:175px;">
                <div style="font-size:1.65rem;margin-bottom:9px;">{icon}</div>
                <div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;margin-bottom:8px;">{title}</div>
                <div style="color:{TEXT_MUTED};font-size:0.9rem;line-height:1.5;">{description}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button(
                label,
                key=f"home_{title}",
                width="stretch",
                on_click=_navigate,
                args=(destination,),
            )

    st.markdown("### Project status")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Raw image pairs", raw_pairs)
    stat2.metric("Prepared test pairs", test_pairs)
    stat3.metric("Compute", "GPU" if torch.cuda.is_available() else "CPU")
    with st.expander("Need the technical details?"):
        st.write("The video tool includes live FPS and quality charts. Image comparison includes heatmaps and detailed metrics. The quality explorer ranks images by sharpness, edge strength, brightness, or blur.")
        if not standard_model:
            st.warning("A dashboard-compatible trained model was not found in the standard location. The pages will open, but enhancement may use fallback weights until the checkpoint is repaired or retrained.")

    # The original, architecture-heavy dashboard content remains below for
    # reference; the task-first landing page above intentionally returns here.
    return

    # ── Hero banner ───────────────────────────────────────────────────────────
    device_label = "CUDA · GPU Accelerated" if torch.cuda.is_available() else "CPU Mode"
    device_color = SUCCESS if torch.cuda.is_available() else ACCENT_AMBER

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(245,158,11,0.06) 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:0;right:0;width:300px;height:300px;
                    background:radial-gradient(circle, {ACCENT_CYAN}0a 0%, transparent 70%);
                    pointer-events:none;"></div>
        <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.15em;
                    color:{ACCENT_CYAN};font-weight:600;margin-bottom:10px;">
            ◈ &nbsp; Computer Vision Pipeline
        </div>
        <h1 style="font-size:2.6rem;margin:0 0 10px;
                   background:linear-gradient(90deg,{ACCENT_CYAN},{ACCENT_AMBER});
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;">
            RailVisionX
        </h1>
        <p style="color:{TEXT_MUTED};font-size:1rem;max-width:560px;margin:0 0 20px;line-height:1.6;">
            Real-time railway surveillance image deblurring using MobileNetV2 GAN architecture
            with comprehensive quality analytics and interactive frame comparison tools.
        </p>
        <div>
            {status_pill(device_label, device_color)}
            {status_pill("MobileNetV2 U-Net", ACCENT_CYAN)}
            {status_pill("FPN Multi-scale", ACCENT_CYAN)}
            {status_pill("7+ Quality Metrics", ACCENT_AMBER)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick stats ───────────────────────────────────────────────────────────
    test_blurred = Path("data/split/test/blurred")
    test_sharp   = Path("data/split/test/sharp")
    num_blurred  = len(list(test_blurred.glob("*.png")) + list(test_blurred.glob("*.jpg"))) if test_blurred.exists() else 0
    num_sharp    = len(list(test_sharp.glob("*.png"))   + list(test_sharp.glob("*.jpg")))   if test_sharp.exists()   else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Blurred Frames", num_blurred, help="Test-split blurred images")
    with c2:
        st.metric("Sharp Frames", num_sharp, help="Test-split sharp ground-truth images")
    with c3:
        st.metric("Dashboard Pages", 3)
    with c4:
        st.metric("Quality Metrics", "7+", help="PSNR, SSIM, MAE, Laplacian, Gradient, Brightness, Pixel Accuracy")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown(f"<h3 style='color:{TEXT_PRIMARY};margin-bottom:16px;'>Available Modules</h3>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🎬", "Real-time Deblurring",
         ["Live webcam & video input", "FPS gauge & inference latency",
          "Live PSNR / SSIM / Sharpness charts", "Snapshot & CSV export",
          "Duration-based auto-stop"],
         ACCENT_CYAN),
        ("🔍", "Compare Frames",
         ["Interactive drag-slider comparison", "Side-by-side grid view",
          "Difference heatmap (7 colormaps)", "Metrics overlay on image",
          "Pixel-level analysis"],
         ACCENT_AMBER),
        ("🏆", "Top-K Rankings",
         ["Multi-metric frame ranking", "Rank-badge gallery view",
          "Plotly score distribution", "Best vs Worst toggle",
          "CSV export of rankings"],
         SUCCESS),
    ]

    for col, (icon, title, bullets, color) in zip([col1, col2, col3], cards):
        bullet_html = "".join(
            f'<li style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:4px;">'
            f'<span style="color:{color};margin-right:6px;">›</span>{b}</li>'
            for b in bullets
        )
        col.markdown(f"""
        <div style="
            background:{BG_CARD};
            border:1px solid {color}33;
            border-radius:14px;
            padding:24px;
            height:100%;
            transition:transform 0.2s ease, box-shadow 0.2s ease;
        ">
            <div style="font-size:1.8rem;margin-bottom:10px;">{icon}</div>
            <h4 style="color:{color};margin-bottom:12px;font-size:1rem;font-weight:600;">{title}</h4>
            <ul style="padding-left:0;list-style:none;margin:0;">
                {bullet_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── System info ───────────────────────────────────────────────────────────
    st.markdown(f"<h3 style='color:{TEXT_PRIMARY};margin-bottom:16px;'>System Info</h3>",
                unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    items = [
        ("Compute Device", "CUDA (GPU)" if torch.cuda.is_available() else "CPU"),
        ("PyTorch Version", torch.__version__),
        ("Model Architecture", "MobileNetV2 U-Net"),
        ("Input Resolution", "256 × 256"),
    ]
    for col, (k, v) in zip([s1, s2, s3, s4], items):
        col.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
                    padding:14px 18px;text-align:center;">
            <div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:6px;">{k}</div>
            <div style="color:{ACCENT_CYAN};font-size:0.95rem;font-weight:600;
                        font-family:'JetBrains Mono',monospace;">{v}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture overview ─────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{TEXT_PRIMARY};margin-bottom:16px;'>Model Architecture</h3>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;
                padding:24px 32px;font-family:'JetBrains Mono',monospace;
                font-size:0.82rem;color:{TEXT_MUTED};line-height:1.8;">
        <span style="color:{ACCENT_CYAN};font-weight:700;">Input</span>
        &nbsp;(Blurred Image 256×256)
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓
        <br><span style="color:{ACCENT_AMBER};font-weight:600;">Deblurring Network</span>
        <br>&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:{ACCENT_CYAN};">MobileNetV2 Backbone</span>
        &nbsp;&nbsp;← Pre-trained encoder · ~3.5M params
        <br>&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:{ACCENT_CYAN};">Feature Pyramid Network</span>
        &nbsp;← Multi-scale feature fusion
        <br>&nbsp;&nbsp;&nbsp;&nbsp;└── <span style="color:{ACCENT_CYAN};">U-Net Decoder</span>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← Transposed convolutions
        <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓
        <br><span style="color:{SUCCESS};font-weight:700;">Output</span>
        &nbsp;(Sharp Image 256×256)
        <br><br>
        <span style="color:{TEXT_MUTED};font-size:0.75rem;">
            Loss = λ<sub>L1</sub>·L1 + λ<sub>perc</sub>·VGG + λ<sub>adv</sub>·Adversarial
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
