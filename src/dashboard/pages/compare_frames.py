"""Compare frames page - Interactive image comparison with multiple visualization modes."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path

from src.dashboard.components.image_compare import (
    image_compare,
    image_compare_grid,
    image_diff_heatmap,
    image_metrics_overlay
)


def main():
    st.set_page_config(page_title="Compare Frames", layout="wide")
    st.title("🔍 Frame Comparison")
    
    st.markdown("""
    Compare blurred and deblurred frames with interactive visualizations.
    Upload images or load from processed results.
    """)
    
    # Sidebar options
    st.sidebar.markdown("## 📂 Image Source")
    source = st.sidebar.radio("Select source:", ["Upload Images", "Test Dataset", "Load from Results"])
    
    img1 = None
    img2 = None
    
    if source == "Upload Images":
        st.sidebar.markdown("### Upload Images")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            file1 = st.file_uploader("Blurred Image", type=['png', 'jpg', 'jpeg'])
        with col2:
            file2 = st.file_uploader("Deblurred Image", type=['png', 'jpg', 'jpeg'])
        
        if file1 and file2:
            # Read uploaded files
            img1_bytes = np.frombuffer(file1.read(), np.uint8)
            img2_bytes = np.frombuffer(file2.read(), np.uint8)
            
            img1 = cv2.imdecode(img1_bytes, cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(img2_bytes, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    elif source == "Test Dataset":
        st.sidebar.markdown("### Load from Test Dataset")
        
        # Load from data/split/test
        blurred_path = Path("data/split/test/blurred")
        sharp_path = Path("data/split/test/sharp")
        
        if blurred_path.exists() and sharp_path.exists():
            blurred_images = sorted(blurred_path.glob("*.png")) + sorted(blurred_path.glob("*.jpg"))
            sharp_images = sorted(sharp_path.glob("*.png")) + sorted(sharp_path.glob("*.jpg"))
            
            if blurred_images and sharp_images:
                st.sidebar.info(f"Found {len(blurred_images)} blurred and {len(sharp_images)} sharp images")
                
                selected_idx = st.sidebar.slider(
                    "Select frame:",
                    0, min(len(blurred_images), len(sharp_images)) - 1, 0
                )
                
                img1 = cv2.imread(str(blurred_images[selected_idx]))
                img2 = cv2.imread(str(sharp_images[selected_idx]))
                
                if img1 is not None and img2 is not None:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    
                    st.sidebar.success(f"Loaded: {blurred_images[selected_idx].name}")
                else:
                    st.error("Failed to load images")
            else:
                st.warning("No images found in test dataset folders")
        else:
            st.error("Test dataset folders not found!")
    
    elif source == "Load from Results":
        st.sidebar.markdown("### Select from Results")
        results_path = Path("outputs/result")
        
        if results_path.exists():
            # List available result folders
            result_folders = [f for f in results_path.iterdir() if f.is_dir()]
            
            if result_folders:
                selected_folder = st.sidebar.selectbox(
                    "Result folder:",
                    result_folders,
                    format_func=lambda x: x.name
                )
                
                # Load images from selected folder
                blurred_path = selected_folder / "blurred"
                sharp_path = selected_folder / "sharp"
                
                if blurred_path.exists() and sharp_path.exists():
                    blurred_images = sorted(blurred_path.glob("*.png")) + sorted(blurred_path.glob("*.jpg"))
                    sharp_images = sorted(sharp_path.glob("*.png")) + sorted(sharp_path.glob("*.jpg"))
                    
                    if blurred_images and sharp_images:
                        selected_idx = st.sidebar.slider(
                            "Select frame:",
                            0, min(len(blurred_images), len(sharp_images)) - 1, 0
                        )
                        
                        img1 = cv2.imread(str(blurred_images[selected_idx]))
                        img2 = cv2.imread(str(sharp_images[selected_idx]))
                        
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                else:
                    st.warning("No blurred/sharp folders found in results")
            else:
                st.info("No result folders found. Process some videos first!")
        else:
            st.info("No results directory found. Run deblurring first!")
    
    if img1 is not None and img2 is not None:
        # Visualization mode selection
        st.sidebar.markdown("## 🎨 Visualization Mode")
        viz_mode = st.sidebar.selectbox(
            "Select mode:",
            ["Interactive Slider", "Side-by-Side Grid", "Difference Heatmap", "Metrics Overlay", "All Views"]
        )
        
        # Display based on selected mode
        if viz_mode == "Interactive Slider":
            st.markdown("### 🎚️ Interactive Slider Comparison")
            st.markdown("*Drag the slider or click to compare images*")
            
            slider_pos = st.sidebar.slider("Starting position (%)", 0, 100, 50)
            show_labels = st.sidebar.checkbox("Show labels", value=True)
            width = st.sidebar.slider("Viewer width (px)", 400, 1200, 700, 50)
            
            image_compare(img1, img2, "Blurred", "Deblurred", width, show_labels, slider_pos)
        
        elif viz_mode == "Side-by-Side Grid":
            st.markdown("### 📊 Side-by-Side Grid")
            image_compare_grid([img1, img2], ["Blurred Input", "Deblurred Output"], columns=2)
        
        elif viz_mode == "Difference Heatmap":
            st.markdown("### 🌡️ Difference Heatmap")
            
            colormap = st.sidebar.selectbox(
                "Colormap:",
                ["hot", "viridis", "plasma", "inferno", "magma", "jet", "coolwarm"]
            )
            
            diff_heatmap = image_diff_heatmap(img1, img2, colormap)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img1, caption="Blurred", use_container_width=True)
            with col2:
                st.image(img2, caption="Deblurred", use_container_width=True)
            
            st.image(diff_heatmap, caption=f"Difference Heatmap ({colormap})", use_container_width=True)
            
            # Stats
            diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)) / 255.0
            st.markdown(f"""
            **Difference Statistics:**
            - Mean Difference: {np.mean(diff):.4f}
            - Max Difference: {np.max(diff):.4f}
            - Std Deviation: {np.std(diff):.4f}
            """)
        
        elif viz_mode == "Metrics Overlay":
            st.markdown("### 📋 Metrics Overlay")
            
            # Calculate some demo metrics
            from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
            
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            # Calculate metric values
            blur1 = variance_of_laplacian(img1_norm)
            grad1 = mean_gradient_magnitude(img1_norm)
            blur2 = variance_of_laplacian(img2_norm)
            grad2 = mean_gradient_magnitude(img2_norm)
            
            improvement = ((blur2 / blur1 - 1) * 100) if blur1 > 0 else 0
            
            metrics1 = {
                "Blur": blur1,
                "Gradient": grad1
            }
            
            metrics2 = {
                "Blur": blur2,
                "Gradient": grad2,
                "Improved": f"{improvement:+.1f}%"
            }
            
            position = st.sidebar.selectbox(
                "Overlay position:",
                ["top-left", "top-right", "bottom-left", "bottom-right"]
            )
            
            img1_overlay = image_metrics_overlay(img1, metrics1, position)
            img2_overlay = image_metrics_overlay(img2, metrics2, position)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img1_overlay, caption="Blurred with Metrics", use_container_width=True)
            with col2:
                st.image(img2_overlay, caption="Deblurred with Metrics", use_container_width=True)
        
        else:  # All Views
            st.markdown("### 📑 All Visualization Modes")
            
            # 1. Interactive Slider
            st.markdown("#### 1️⃣ Interactive Slider")
            image_compare(img1, img2, "Blurred", "Deblurred", 700, True, 50)
            
            st.markdown("---")
            
            # 2. Grid
            st.markdown("#### 2️⃣ Side-by-Side Grid")
            image_compare_grid([img1, img2], ["Blurred", "Deblurred"], columns=2)
            
            st.markdown("---")
            
            # 3. Heatmap
            st.markdown("#### 3️⃣ Difference Heatmap")
            diff_heatmap = image_diff_heatmap(img1, img2, 'hot')
            st.image(diff_heatmap, caption="Difference Heatmap", use_container_width=True)
            
            st.markdown("---")
            
            # 4. Metrics Overlay
            st.markdown("#### 4️⃣ Metrics Overlay")
            from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
            
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            metrics = {
                "Blur": variance_of_laplacian(img2_norm),
                "Gradient": mean_gradient_magnitude(img2_norm)
            }
            
            img2_overlay = image_metrics_overlay(img2, metrics, "top-left")
            st.image(img2_overlay, caption="Deblurred with Metrics", use_container_width=True)
    
    else:
        st.info("👆 Select an image source from the sidebar to begin comparison")


if __name__ == "__main__":
    main()
