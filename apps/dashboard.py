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
    if page == "🏠  Home":
        show_home()
    elif page == "🎬  Enhance a video":
        from src.dashboard.pages.realtime import main as realtime_main
        realtime_main()
    elif page == "🔍  Compare two images":
        from src.dashboard.pages.compare_frames import main as compare_main
        compare_main()
    elif page == "📊  Explore image quality":
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


if __name__ == "__main__":
    main()
