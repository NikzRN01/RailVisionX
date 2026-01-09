
"""
RailVisionX Dashboard - Main Entry Point
Multi-page Streamlit application for real-time deblurring analysis
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="RailVisionX Dashboard",
        page_icon="🚆",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Sidebar navigation
    st.sidebar.title("🚆 RailVisionX")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Real-time Deblurring", "Compare Frames", "Top-K Rankings"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Real-time Deblurring":
        show_realtime()
    elif page == "Compare Frames":
        show_compare()
    elif page == "Top-K Rankings":
        show_topk()


def show_home():
    """Home page with overview and quick stats."""
    st.title("🚆 RailVisionX Dashboard")
    st.markdown("### Welcome to the Railway Image Deblurring Analysis Platform")
    
    st.markdown("""
    This dashboard provides comprehensive tools for analyzing and improving railway surveillance image quality.
    """)
    
    st.markdown("---")
    
    # Feature cards
    st.markdown("### 📋 Available Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎬 Real-time Deblurring**
        
        - Live webcam & video input
        - Comprehensive metrics display
        - FPS performance tracking
        - Snapshot & CSV export
        - Duration-based auto-stop
        """)
    
    with col2:
        st.markdown("""
        **🔍 Compare Frames**
        
        - Interactive slider control
        - Grid & heatmap views
        - Metrics overlay option
        - Multiple data sources
        - Pixel-level analysis
        """)
    
    with col3:
        st.markdown("""
        **🏆 Top-K Rankings**
        
        - Quality frame ranking
        - Multiple metrics analysis
        - Detailed visualizations
        - CSV export capability
        - Distribution charts
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Count test images
    test_blurred = Path("data/split/test/blurred")
    test_sharp = Path("data/split/test/sharp")
    
    num_blurred = len(list(test_blurred.glob("*.png"))) + len(list(test_blurred.glob("*.jpg"))) if test_blurred.exists() else 0
    num_sharp = len(list(test_sharp.glob("*.png"))) + len(list(test_sharp.glob("*.jpg"))) if test_sharp.exists() else 0
    
    with col1:
        st.metric("Blurred Test Images", num_blurred)
    with col2:
        st.metric("Sharp Test Images", num_sharp)
    with col3:
        st.metric("Dashboard Pages", 3)
    with col4:
        st.metric("Metrics Available", "7+")
    
    st.markdown("---")
    
    # System Info
    st.markdown("### 🔧 System Info")
    
    import torch
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Compute Device:** {device}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
    with col2:
        st.write(f"**Model Architecture:** MobileNetV2 U-Net")
        st.write(f"**Input Resolution:** 256×256")


def show_realtime():
    """Load and display real-time deblurring page."""
    from src.dashboard.pages.realtime import main as realtime_main
    realtime_main()


def show_compare():
    """Load and display compare frames page."""
    from src.dashboard.pages.compare_frames import main as compare_main
    compare_main()


def show_topk():
    """Load and display top-k rankings page."""
    from src.dashboard.pages.top10 import main as topk_main
    topk_main()


if __name__ == "__main__":
    main()
