"""Top-K frame rankings page — Enhanced with rank badges, Plotly charts, and styled gallery."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from PIL import Image

from src.analytics.topk_frames import topk_indices
from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
from src.quality.lowlight_score import mean_intensity
from src.dashboard.theme import (
    inject_theme, section_header,
    ACCENT_CYAN, ACCENT_AMBER, SUCCESS, DANGER,
    TEXT_MUTED, TEXT_PRIMARY, BG_CARD, BG_DEEP, BORDER,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RANK_COLORS   = ["#FFD700", "#C0C0C0", "#CD7F32"]   # gold, silver, bronze
RANK_LABELS   = ["🥇", "🥈", "🥉"]
_PLOTLY       = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_images(folder: Path, max_n: int = 100):
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    images = []
    for p in files[:max_n]:
        img = cv2.imread(str(p))
        if img is not None:
            images.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), p.name))
    return images


def score_frame(img: np.ndarray, metric: str) -> float:
    n = img.astype(np.float32) / 255.0
    if metric == "Sharpness (Laplacian)":
        return variance_of_laplacian(n)
    elif metric == "Edge Strength (Gradient)":
        return mean_gradient_magnitude(n)
    elif metric == "Brightness":
        g = cv2.cvtColor((n * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        return mean_intensity(g)
    elif metric == "Blur (inverse)":
        return -variance_of_laplacian(n)
    return 0.0


def _np_to_b64(img: np.ndarray, size: int = 200) -> str:
    pil = Image.fromarray(img)
    pil.thumbnail((size, size))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _rank_badge_html(rank: int, name: str, score: float, img_b64: str) -> str:
    """Return an HTML card with rank badge overlaid on thumbnail."""
    if rank <= 3:
        badge_color = RANK_COLORS[rank - 1]
        badge_icon  = RANK_LABELS[rank - 1]
    else:
        badge_color = ACCENT_CYAN
        badge_icon  = f"#{rank}"

    return f"""
    <div style="
        background:{BG_CARD};
        border:1px solid {badge_color}44;
        border-radius:12px;
        overflow:hidden;
        transition:transform 0.2s ease,box-shadow 0.2s ease;
        cursor:default;
    " onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 32px {badge_color}33';"
       onmouseout="this.style.transform='none';this.style.boxShadow='none';">
        <div style="position:relative;">
            <img src="data:image/png;base64,{img_b64}"
                 style="width:100%;display:block;border-radius:12px 12px 0 0;object-fit:cover;max-height:160px;">
            <div style="
                position:absolute;top:8px;left:8px;
                background:{badge_color};
                color:#000;font-weight:700;font-size:0.85rem;
                padding:3px 10px;border-radius:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.4);
            ">{badge_icon}</div>
        </div>
        <div style="padding:10px 12px;">
            <div style="color:{TEXT_PRIMARY};font-size:0.8rem;font-weight:600;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                 title="{name}">{name}</div>
            <div style="color:{badge_color};font-size:0.85rem;font-weight:700;
                        font-family:'JetBrains Mono',monospace;margin-top:3px;">
                {score:.5f}
            </div>
        </div>
    </div>"""


def _dist_chart(scores: list, top_indices_set: set, k: int, metric: str) -> go.Figure:
    all_s = sorted(scores)
    top_s = sorted([scores[i] for i in top_indices_set])
    rest  = sorted([s for i, s in enumerate(scores) if i not in top_indices_set])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rest,  name="Other frames",
                               marker_color=TEXT_MUTED + "88",
                               opacity=0.7, nbinsx=25))
    fig.add_trace(go.Histogram(x=top_s, name=f"Top-{k} frames",
                               marker_color=ACCENT_CYAN + "cc",
                               opacity=0.9, nbinsx=12))
    fig.update_layout(
        **_PLOTLY,
        barmode="overlay",
        height=320,
        xaxis_title=metric,
        yaxis_title="Count",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
    return fig


def _score_timeline(scores: list, top_set: set) -> go.Figure:
    xs = list(range(len(scores)))
    colors = [ACCENT_CYAN if i in top_set else TEXT_MUTED + "66" for i in xs]
    fig = go.Figure(go.Bar(
        x=xs, y=scores, marker_color=colors,
        hovertemplate="Frame %{x}<br>Score: %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(**_PLOTLY, height=250,
                      yaxis_title="Score", xaxis_title="Frame Index",
                      showlegend=False)
    return fig


# ── Main page ─────────────────────────────────────────────────────────────────
def main():
    inject_theme()

    st.markdown(f"""
    <h1 style="margin-bottom:4px;">🏆 Top-K Frame Rankings</h1>
    <p style="color:{TEXT_MUTED};margin-bottom:24px;">
        Rank and inspect frames by quality metric · Identify best and worst captures
    </p>
    """, unsafe_allow_html=True)

    # ── Sidebar config ────────────────────────────────────────────────────────
    st.sidebar.markdown(f"<h4 style='color:{ACCENT_CYAN};'>⚙ Configuration</h4>",
                        unsafe_allow_html=True)
    source = st.sidebar.radio("Data source", ["Test Dataset", "Upload Images"],
                              label_visibility="collapsed")
    metric = st.sidebar.selectbox(
        "Ranking metric",
        ["Sharpness (Laplacian)", "Edge Strength (Gradient)", "Brightness", "Blur (inverse)"],
    )
    k = st.sidebar.slider("Top-K (frames to select)", 3, 50, 10, 1)
    ranking_mode = st.sidebar.radio("Mode", ["Best (Highest)", "Worst (Lowest)"],
                                    label_visibility="visible")
    largest = ranking_mode == "Best (Highest)"

    images = []

    if source == "Test Dataset":
        st.sidebar.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.8rem;margin-top:8px;'>Test split folder</div>",
                            unsafe_allow_html=True)
        dtype   = st.sidebar.radio("Type", ["Blurred", "Sharp"])
        folder  = Path(f"data/split/test/{dtype.lower()}")
        max_n   = st.sidebar.slider("Max images", 10, 200, 50, 10)
        if folder.exists():
            with st.spinner("Loading images…"):
                images = load_images(folder, max_n)
            st.sidebar.success(f"✅ {len(images)} images loaded")
        else:
            st.error(f"Folder not found: `{folder}`")

    else:
        uploaded = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"],
                                    accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                arr = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f.name))
            st.sidebar.success(f"✅ {len(images)} uploaded")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if not images:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:2px dashed {BORDER};border-radius:16px;
                    padding:60px;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🖼</div>
            <div style="color:{TEXT_MUTED};font-size:1rem;">
                Select a data source from the sidebar to begin ranking
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner(f"Scoring {len(images)} frames on **{metric}**…"):
        scores = [score_frame(img, metric) for img, _ in images]

    top_idx  = topk_indices(scores, k, largest=largest)
    top_set  = set(top_idx)

    # Summary KPI strip
    best_score = scores[top_idx[0]] if top_idx else 0
    avg_score  = float(np.mean(scores))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Frames", len(images))
    c2.metric(f"{'Best' if largest else 'Worst'} Score", f"{best_score:.5f}")
    c3.metric("Average Score", f"{avg_score:.5f}")
    c4.metric("Top-K Selected", k)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── View tabs ─────────────────────────────────────────────────────────────
    tab_gallery, tab_detail, tab_dist = st.tabs(
        ["🖼  Gallery", "🔎  Detail View", "📈  Score Distribution"]
    )

    # ── Gallery tab ───────────────────────────────────────────────────────────
    with tab_gallery:
        section_header(
            f"Top {k} Frames · {ranking_mode}",
            f"Ranked by {metric} · Hover for effect",
        )
        cols_per_row = st.slider("Columns", 2, 6, 4, key="gallery_cols")
        cols = st.columns(cols_per_row)
        for rank_idx, frame_idx in enumerate(top_idx):
            img, name = images[frame_idx]
            b64 = _np_to_b64(img)
            col = cols[rank_idx % cols_per_row]
            col.markdown(
                _rank_badge_html(rank_idx + 1, name, scores[frame_idx], b64),
                unsafe_allow_html=True,
            )

    # ── Detail tab ────────────────────────────────────────────────────────────
    with tab_detail:
        results_df = pd.DataFrame([
            {"Rank": r + 1, "File": images[i][1], "Score": scores[i]}
            for r, i in enumerate(top_idx)
        ])
        selected = st.selectbox(
            "Select frame to inspect",
            range(1, len(top_idx) + 1),
            format_func=lambda x: f"#{x} — {results_df.loc[x-1,'File']} (score: {results_df.loc[x-1,'Score']:.5f})",
        )
        fi  = top_idx[selected - 1]
        img = images[fi][0]

        ic, sc = st.columns([3, 2])
        with ic:
            st.image(img, caption=f"Rank #{selected} · {images[fi][1]}",
                     use_container_width=True)
        with sc:
            section_header("Frame Details")
            img_norm = img.astype(np.float32) / 255.0
            gray     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            det = {
                "File":       images[fi][1],
                "Rank":       f"#{selected} of {k}",
                "Score":      f"{scores[fi]:.6f}",
                "Sharpness":  f"{variance_of_laplacian(img_norm):.5f}",
                "Gradient":   f"{mean_gradient_magnitude(img_norm):.5f}",
                "Brightness": f"{mean_intensity(gray):.5f}",
                "Shape":      f"{img.shape[1]}×{img.shape[0]}",
            }
            for key, val in det.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            border-bottom:1px solid {BORDER};padding:7px 0;">
                    <span style="color:{TEXT_MUTED};font-size:0.82rem;">{key}</span>
                    <span style="color:{ACCENT_CYAN};font-size:0.85rem;
                                 font-family:'JetBrains Mono',monospace;font-weight:600;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        # Ranking table
        st.markdown(f"<h4 style='color:{TEXT_MUTED};margin-top:24px;'>Full Rankings Table</h4>",
                    unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        if st.button("📥 Export CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button("⬇ Download", csv,
                               f"topk_{metric.replace(' ','_').lower()}.csv", "text/csv")

    # ── Distribution tab ──────────────────────────────────────────────────────
    with tab_dist:
        section_header("📈 Score Distribution", "Top-K frames highlighted in cyan")

        st.plotly_chart(_dist_chart(scores, top_set, k, metric),
                        use_container_width=True)

        st.markdown(f"<h4 style='color:{TEXT_MUTED};margin-top:8px;'>Score Timeline</h4>",
                    unsafe_allow_html=True)
        st.plotly_chart(_score_timeline(scores, top_set), use_container_width=True)

        # Stats comparison
        top_scores  = [scores[i] for i in top_idx]
        rest_scores = [scores[i] for i in range(len(scores)) if i not in top_set]

        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"<h5 style='color:{ACCENT_CYAN};'>All Frames</h5>", unsafe_allow_html=True)
            for label, val in [
                ("Mean",    np.mean(scores)),
                ("Std Dev", np.std(scores)),
                ("Min",     np.min(scores)),
                ("Max",     np.max(scores)),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            border-bottom:1px solid {BORDER};padding:6px 0;">
                    <span style="color:{TEXT_MUTED};font-size:0.82rem;">{label}</span>
                    <span style="color:{TEXT_PRIMARY};font-family:'JetBrains Mono',monospace;
                                 font-size:0.85rem;">{val:.5f}</span>
                </div>""", unsafe_allow_html=True)

        with s2:
            st.markdown(f"<h5 style='color:{ACCENT_CYAN};'>Top-{k} Frames</h5>", unsafe_allow_html=True)
            for label, val in [
                ("Mean",    np.mean(top_scores)),
                ("Std Dev", np.std(top_scores)),
                ("Min",     np.min(top_scores)),
                ("Max",     np.max(top_scores)),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            border-bottom:1px solid {BORDER};padding:6px 0;">
                    <span style="color:{TEXT_MUTED};font-size:0.82rem;">{label}</span>
                    <span style="color:{SUCCESS};font-family:'JetBrains Mono',monospace;
                                 font-size:0.85rem;font-weight:600;">{val:.5f}</span>
                </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
