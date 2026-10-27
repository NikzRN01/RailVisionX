"""Compare frames page — Enhanced UI with Plotly diff heatmap and styled controls."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from src.dashboard.components.image_compare import (
    image_compare,
    image_compare_grid,
    image_diff_heatmap,
    image_metrics_overlay,
)
from src.dashboard.theme import (
    inject_theme, section_header,
    ACCENT_CYAN, ACCENT_AMBER, SUCCESS, DANGER,
    TEXT_MUTED, TEXT_PRIMARY, BG_CARD, BG_DEEP, BORDER,
)

# Inline plotly layout since theme doesn't export it
_PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showgrid=True),
)


def _stat_card(label: str, value: str, color: str = ACCENT_CYAN) -> str:
    return f"""
    <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
                padding:14px 18px;text-align:center;">
        <div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;">{label}</div>
        <div style="color:{color};font-size:1.1rem;font-weight:700;
                    font-family:'JetBrains Mono',monospace;">{value}</div>
    </div>"""


def _diff_plotly(img1: np.ndarray, img2: np.ndarray) -> go.Figure:
    """Return a Plotly heatmap of per-channel mean absolute difference."""
    if img1.dtype == np.uint8:
        a = img1.astype(np.float32) / 255.0
    else:
        a = img1.astype(np.float32)
    if img2.dtype == np.uint8:
        b = img2.astype(np.float32) / 255.0
    else:
        b = img2.astype(np.float32)

    if a.shape != b.shape:
        b = cv2.resize((b * 255).astype(np.uint8), (a.shape[1], a.shape[0])) / 255.0

    diff = np.mean(np.abs(a - b), axis=2)  # HxW grayscale difference

    fig = go.Figure(go.Heatmap(
        z=diff,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=TEXT_MUTED, size=10),
            title=dict(text="Δ Pixel", font=dict(color=TEXT_MUTED, size=10)),
        ),
    ))
    fig.update_layout(
        **_PLOTLY,
        title=dict(text="Pixel-level Difference Heatmap", font=dict(color=TEXT_MUTED, size=12)),
        height=400,
        yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


def main():
    inject_theme()

    st.markdown(f"""
    <h1 style="margin-bottom:4px;">🔍 Compare two images</h1>
    <p style="color:{TEXT_MUTED};margin-bottom:24px;">
        Upload two images or use the prepared test set to see a clear before / after view.
    </p>
    """, unsafe_allow_html=True)

    # ── Sidebar: source ───────────────────────────────────────────────────────
    st.sidebar.markdown(f"<h4 style='color:{ACCENT_CYAN};'>📂 Image Source</h4>",
                        unsafe_allow_html=True)
    source = st.sidebar.radio("Source", ["Upload Images", "Test Dataset", "Load from Results"],
                              label_visibility="collapsed")

    img1 = img2 = None

    # ── Image loading ─────────────────────────────────────────────────────────
    if source == "Upload Images":
        st.sidebar.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.8rem;'>Upload a blurred and a deblurred image.</div>",
                            unsafe_allow_html=True)
        col_u1, col_u2 = st.sidebar.columns(2)
        file1 = col_u1.file_uploader("Blurred",   type=["png", "jpg", "jpeg"], key="f1")
        file2 = col_u2.file_uploader("Deblurred", type=["png", "jpg", "jpeg"], key="f2")

        if file1 and file2:
            img1 = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            img2 = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )

    elif source == "Test Dataset":
        blurred_path = Path("data/split/test/blurred")
        sharp_path   = Path("data/split/test/sharp")
        if blurred_path.exists() and sharp_path.exists():
            blurred_imgs = sorted(blurred_path.glob("*.png")) + sorted(blurred_path.glob("*.jpg"))
            sharp_imgs   = sorted(sharp_path.glob("*.png"))   + sorted(sharp_path.glob("*.jpg"))
            if blurred_imgs and sharp_imgs:
                st.sidebar.info(f"Found {len(blurred_imgs)} pairs")
                idx = st.sidebar.slider("Frame index", 0,
                                        min(len(blurred_imgs), len(sharp_imgs)) - 1, 0)
                img1 = cv2.cvtColor(cv2.imread(str(blurred_imgs[idx])), cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(cv2.imread(str(sharp_imgs[idx])),   cv2.COLOR_BGR2RGB)
                st.sidebar.success(f"📄 {blurred_imgs[idx].name}")
            else:
                st.warning("No images found in test dataset.")
        else:
            st.error("Test dataset folders not found.")

    elif source == "Load from Results":
        results_path = Path("outputs/result")
        if results_path.exists():
            folders = [f for f in results_path.iterdir() if f.is_dir()]
            if folders:
                sel = st.sidebar.selectbox("Result folder", folders,
                                           format_func=lambda x: x.name)
                bp = sel / "blurred"
                sp = sel / "sharp"
                if bp.exists() and sp.exists():
                    bi = sorted(bp.glob("*.png")) + sorted(bp.glob("*.jpg"))
                    si = sorted(sp.glob("*.png")) + sorted(sp.glob("*.jpg"))
                    if bi and si:
                        idx = st.sidebar.slider("Frame", 0, min(len(bi), len(si)) - 1, 0)
                        img1 = cv2.cvtColor(cv2.imread(str(bi[idx])), cv2.COLOR_BGR2RGB)
                        img2 = cv2.cvtColor(cv2.imread(str(si[idx])), cv2.COLOR_BGR2RGB)
                else:
                    st.warning("No blurred/sharp folders in result.")
            else:
                st.info("No result folders found. Process some videos first.")
        else:
            st.info("No results directory. Run deblurring first.")

    # ── Visualisation ─────────────────────────────────────────────────────────
    if img1 is not None and img2 is not None:
        # Compute diff stats eagerly (used in multiple modes)
        diff_f = np.abs(img1.astype(np.float32) - img2.astype(np.float32)) / 255.0

        # Quality metrics
        from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        n1 = img1.astype(np.float32) / 255.0
        n2 = img2.astype(np.float32) / 255.0
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        lap1  = variance_of_laplacian(n1)
        lap2  = variance_of_laplacian(n2)
        grad1 = mean_gradient_magnitude(n1)
        grad2 = mean_gradient_magnitude(n2)
        ssim_score = ssim_fn(g1, g2, data_range=1.0)
        psnr_score = psnr_fn(n1, n2, data_range=1.0)
        mae_score  = float(np.mean(diff_f))

        # ── Summary stats strip ───────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        stat_pairs = [
            ("PSNR",           f"{psnr_score:.2f} dB", ACCENT_CYAN),
            ("SSIM",           f"{ssim_score:.4f}",     ACCENT_CYAN),
            ("MAE",            f"{mae_score:.4f}",      ACCENT_AMBER),
            ("Sharp ↑ Blur",   f"{lap1:.4f}",           TEXT_MUTED),
            ("Sharp ↑ Deblu",  f"{lap2:.4f}",           SUCCESS),
        ]
        for col, (lbl, val, color) in zip([c1, c2, c3, c4, c5], stat_pairs):
            col.markdown(_stat_card(lbl, val, color), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Viz mode ──────────────────────────────────────────────────────────
        st.sidebar.markdown(f"<h4 style='color:{ACCENT_CYAN};'>🎨 Visualisation Mode</h4>",
                            unsafe_allow_html=True)
        viz_mode = st.sidebar.selectbox(
            "Mode",
            ["Interactive Slider", "Side-by-Side Grid", "Plotly Heatmap",
             "CV Heatmap", "Metrics Overlay", "All Views"],
            label_visibility="collapsed",
        )

        if viz_mode == "Interactive Slider":
            section_header("🎚 Drag-Slider Comparison",
                           "Click anywhere or drag the handle to reveal before/after")
            slider_pos  = st.sidebar.slider("Start position (%)", 0, 100, 50)
            show_labels = st.sidebar.checkbox("Show labels", value=True)
            width       = st.sidebar.slider("Viewer width (px)", 400, 1400, 800, 50)
            image_compare(img1, img2, "Blurred", "Deblurred", width, show_labels, slider_pos)

        elif viz_mode == "Side-by-Side Grid":
            section_header("📊 Side-by-Side")
            image_compare_grid([img1, img2], ["Blurred Input", "Deblurred Output"], columns=2)

        elif viz_mode == "Plotly Heatmap":
            section_header("🌡 Interactive Plotly Heatmap",
                           "Zoom, pan, and hover to inspect pixel-level differences")
            st.plotly_chart(_diff_plotly(img1, img2), width="stretch")

            s1, s2, s3 = st.columns(3)
            s1.markdown(_stat_card("Mean Δ",  f"{np.mean(diff_f):.4f}"), unsafe_allow_html=True)
            s2.markdown(_stat_card("Max Δ",   f"{np.max(diff_f):.4f}",  DANGER), unsafe_allow_html=True)
            s3.markdown(_stat_card("Std Dev", f"{np.std(diff_f):.4f}",  ACCENT_AMBER), unsafe_allow_html=True)

        elif viz_mode == "CV Heatmap":
            section_header("🌡 OpenCV Heatmap")
            colormap = st.sidebar.selectbox(
                "Colormap",
                ["hot", "viridis", "plasma", "inferno", "magma", "jet", "coolwarm"],
            )
            diff_heatmap = image_diff_heatmap(img1, img2, colormap)
            r1, r2, r3 = st.columns(3)
            r1.image(img1, caption="Blurred", width="stretch")
            r2.image(img2, caption="Deblurred", width="stretch")
            r3.image(diff_heatmap, caption=f"Heatmap ({colormap})", width="stretch")

        elif viz_mode == "Metrics Overlay":
            section_header("📋 Metrics Overlay",
                           "Quality values burned onto each image")
            position = st.sidebar.selectbox("Overlay position",
                                            ["top-left", "top-right", "bottom-left", "bottom-right"])
            imp = ((lap2 / lap1 - 1) * 100) if lap1 > 0 else 0
            ov1 = image_metrics_overlay(img1, {"Blur": lap1, "Grad": grad1}, position)
            ov2 = image_metrics_overlay(img2, {"Blur": lap2, "Grad": grad2, "Imp": f"{imp:+.1f}%"}, position)
            c_l, c_r = st.columns(2)
            c_l.image(ov1, caption="Blurred + Metrics", width="stretch")
            c_r.image(ov2, caption="Deblurred + Metrics", width="stretch")

        else:  # All Views
            section_header("📑 All Visualisations")

            st.markdown(f"<h4 style='color:{TEXT_MUTED};'>① Interactive Slider</h4>",
                        unsafe_allow_html=True)
            image_compare(img1, img2, "Blurred", "Deblurred", 800, True, 50)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{TEXT_MUTED};'>② Plotly Difference Heatmap</h4>",
                        unsafe_allow_html=True)
            st.plotly_chart(_diff_plotly(img1, img2), width="stretch")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{TEXT_MUTED};'>③ Side-by-Side</h4>",
                        unsafe_allow_html=True)
            image_compare_grid([img1, img2], ["Blurred", "Deblurred"], columns=2)

    else:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:2px dashed {BORDER};border-radius:16px;
                    padding:60px;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🖼</div>
            <div style="color:{TEXT_MUTED};font-size:1rem;">
                Select an image source from the sidebar to begin comparison
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
