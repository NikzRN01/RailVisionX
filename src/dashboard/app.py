"""RailVisionX Dashboard - src-level app (delegates to apps/dashboard.py entry point)."""

# This file is a thin redirect. The actual dashboard lives in apps/dashboard.py
# Run with: streamlit run apps/dashboard.py
import streamlit as st
from src.dashboard.theme import inject_theme


def main():
    inject_theme()
    st.info("Run the dashboard via: `streamlit run apps/dashboard.py`")


if __name__ == "__main__":
    main()
