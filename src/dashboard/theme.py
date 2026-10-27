"""RailVisionX global dark-theme CSS for Streamlit.

Call inject_theme() once at the top of every page / app entry point.
"""

import streamlit as st

# ── colour palette ────────────────────────────────────────────────────────────
BG_DEEP      = "#0a0e1a"      # page background
BG_CARD      = "rgba(255,255,255,0.05)"  # glassmorphism card
ACCENT_CYAN  = "#00d4ff"      # sharpness / primary accent
ACCENT_AMBER = "#f59e0b"      # warning / highlight
SUCCESS      = "#10b981"      # improvement green
DANGER       = "#ef4444"      # degradation red
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED   = "#94a3b8"
BORDER       = "rgba(0,212,255,0.15)"


def inject_theme() -> None:
    """Inject the RailVisionX dark theme CSS into the Streamlit page."""
    css = f"""
    <style>
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root / page background ──────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {{
        background-color: {BG_DEEP} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Main content area */
    [data-testid="stMain"], .main .block-container {{
        background-color: {BG_DEEP} !important;
        padding-top: 1.5rem !important;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d1225 0%, #0a0e1a 100%) !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: {TEXT_MUTED} !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }}

    /* ── Typography ──────────────────────────────────────── */
    h1 {{ color: {ACCENT_CYAN} !important; font-weight: 700 !important;
          letter-spacing: -0.5px !important; }}
    h2, h3 {{ color: {TEXT_PRIMARY} !important; font-weight: 600 !important; }}
    h4, h5, h6 {{ color: {TEXT_MUTED} !important; font-weight: 500 !important; }}
    p, li, span {{ color: {TEXT_PRIMARY} !important; }}
    code, pre {{ font-family: 'JetBrains Mono', monospace !important; }}

    /* ── Metric cards (st.metric) ────────────────────────── */
    [data-testid="stMetric"] {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        backdrop-filter: blur(8px) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,212,255,0.12) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {ACCENT_CYAN} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT_CYAN}22, {ACCENT_CYAN}11) !important;
        border: 1px solid {ACCENT_CYAN}55 !important;
        color: {ACCENT_CYAN} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.03em !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {ACCENT_CYAN}44, {ACCENT_CYAN}22) !important;
        box-shadow: 0 4px 16px {ACCENT_CYAN}33 !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── Progress bar ────────────────────────────────────── */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {ACCENT_CYAN}, {ACCENT_AMBER}) !important;
        border-radius: 4px !important;
    }}

    /* ── Selectbox / Radio / Slider ──────────────────────── */
    [data-testid="stSelectbox"] > div > div {{
        background: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .stRadio > div {{
        gap: 6px !important;
    }}
    .stSlider [data-testid="stSlider"] > div {{
        color: {ACCENT_CYAN} !important;
    }}

    /* ── Info / Success / Warning / Error banners ────────── */
    [data-testid="stAlert"] {{
        border-radius: 10px !important;
        border-left-width: 3px !important;
    }}
    .stSuccess {{ border-left-color: {SUCCESS} !important; background: {SUCCESS}18 !important; }}
    .stWarning {{ border-left-color: {ACCENT_AMBER} !important; background: {ACCENT_AMBER}18 !important; }}
    .stError   {{ border-left-color: {DANGER} !important; background: {DANGER}18 !important; }}
    .stInfo    {{ border-left-color: {ACCENT_CYAN} !important; background: {ACCENT_CYAN}18 !important; }}

    /* ── Dataframe / Table ───────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid {BORDER} !important;
    }}
    .stDataFrame table {{ background: {BG_DEEP} !important; }}
    .stDataFrame th {{
        background: {ACCENT_CYAN}22 !important;
        color: {ACCENT_CYAN} !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }}
    .stDataFrame td {{ color: {TEXT_PRIMARY} !important; }}

    /* ── File uploader ───────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed {ACCENT_CYAN}44 !important;
        border-radius: 12px !important;
        background: {ACCENT_CYAN}08 !important;
    }}

    /* ── Expander ────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        background: {BG_CARD} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {ACCENT_CYAN} !important;
        font-weight: 600 !important;
    }}

    /* ── Horizontal rule ─────────────────────────────────── */
    hr {{
        border: none !important;
        border-top: 1px solid {BORDER} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG_DEEP}; }}
    ::-webkit-scrollbar-thumb {{ background: {ACCENT_CYAN}44; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_CYAN}88; }}

    /* ── Spinner ─────────────────────────────────────────── */
    .stSpinner > div {{ border-top-color: {ACCENT_CYAN} !important; }}

    /* ── Image captions ──────────────────────────────────── */
    [data-testid="caption"] {{ color: {TEXT_MUTED} !important; font-size: 0.78rem !important; }}

    /* ── Checkbox ────────────────────────────────────────── */
    .stCheckbox > label > span {{ color: {TEXT_PRIMARY} !important; }}

    /* ── Top-bar hide ────────────────────────────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", delta_positive: bool = True,
                icon: str = "") -> str:
    """Return an HTML string for a styled glassmorphism metric card."""
    delta_color = SUCCESS if delta_positive else DANGER
    delta_html = ""
    if delta:
        arrow = "▲" if delta_positive else "▼"
        delta_html = f"""
        <div style="color:{delta_color};font-size:0.82rem;font-weight:600;margin-top:4px;">
            {arrow} {delta}
        </div>"""
    return f"""
    <div style="
        background:{BG_CARD};
        border:1px solid {BORDER};
        border-radius:12px;
        padding:16px 20px;
        backdrop-filter:blur(8px);
        transition:transform 0.2s ease;
        text-align:center;
    ">
        <div style="font-size:1.4rem;margin-bottom:4px;">{icon}</div>
        <div style="color:{TEXT_MUTED};font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;">{label}</div>
        <div style="color:{ACCENT_CYAN};font-size:1.7rem;font-weight:700;
                    font-family:'JetBrains Mono',monospace;">{value}</div>
        {delta_html}
    </div>"""


def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    sub_html = f'<p style="color:{TEXT_MUTED};font-size:0.9rem;margin-top:-8px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <h3 style="color:{TEXT_PRIMARY};margin-bottom:2px;">{title}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def status_pill(label: str, color: str = ACCENT_CYAN) -> str:
    """Return an inline HTML status pill."""
    return f"""<span style="
        background:{color}22;
        color:{color};
        border:1px solid {color}55;
        border-radius:20px;
        padding:3px 12px;
        font-size:0.78rem;
        font-weight:600;
        letter-spacing:0.05em;
        display:inline-block;
        margin:3px 4px;
    ">{label}</span>"""
