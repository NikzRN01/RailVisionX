"""Top-10 frames page - Display best/worst frames ranked by quality metrics."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

from src.analytics.topk_frames import topk_indices, FrameScore
from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
from src.quality.lowlight_score import mean_intensity
from src.dashboard.components.image_compare import image_compare, image_compare_grid


def load_images_from_folder(folder_path: Path, max_images: int = 100):
    """Load images from a folder with limit."""
    images = []
    image_files = sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpg"))
    
    for img_path in image_files[:max_images]:
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((img, img_path.name))
    
    return images


def calculate_frame_score(img: np.ndarray, metric_type: str) -> float:
    """Calculate quality score for a frame."""
    img_norm = img.astype(np.float32) / 255.0
    
    if metric_type == "Sharpness (Laplacian)":
        return variance_of_laplacian(img_norm)
    elif metric_type == "Edge Strength (Gradient)":
        return mean_gradient_magnitude(img_norm)
    elif metric_type == "Brightness":
        gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        return mean_intensity(gray)
    elif metric_type == "Blur (inverse)":
        return -variance_of_laplacian(img_norm)  # Negative for worst ranking
    else:
        return 0.0


def main():
    st.title("🏆 Top-K Frame Rankings")
    
    st.markdown("""
    Analyze and rank frames by quality metrics to identify best/worst frames.
    Useful for quality assessment and dataset curation.
    """)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Data source
    source = st.sidebar.radio("Data source:", ["Test Dataset", "Upload Folder"])
    
    # Metric selection
    metric_type = st.sidebar.selectbox(
        "Ranking metric:",
        ["Sharpness (Laplacian)", "Edge Strength (Gradient)", "Brightness", "Blur (inverse)"]
    )
    
    # Top-K selection
    k = st.sidebar.slider("Number of frames (K):", 5, 50, 10, 5)
    
    # Ranking mode
    ranking_mode = st.sidebar.radio("Ranking mode:", ["Best (Highest scores)", "Worst (Lowest scores)"])
    
    images = []
    
    if source == "Test Dataset":
        st.sidebar.markdown("### Test Dataset")
        dataset_type = st.sidebar.radio("Dataset type:", ["Blurred", "Sharp"])
        
        folder_path = Path(f"data/split/test/{dataset_type.lower()}")
        
        if folder_path.exists():
            max_images = st.sidebar.slider("Max images to analyze:", 10, 200, 50, 10)
            
            with st.spinner(f"Loading images from {folder_path}..."):
                images = load_images_from_folder(folder_path, max_images)
            
            st.sidebar.success(f"Loaded {len(images)} images")
        else:
            st.error(f"Folder not found: {folder_path}")
    
    elif source == "Upload Folder":
        st.sidebar.markdown("### Upload Images")
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for idx, file in enumerate(uploaded_files):
                img_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append((img, file.name))
            
            st.sidebar.success(f"Uploaded {len(images)} images")
    
    # Process images if available
    if images:
        st.markdown(f"### 📊 Analyzing {len(images)} frames with **{metric_type}**")
        
        # Calculate scores
        with st.spinner("Calculating quality scores..."):
            scores = [calculate_frame_score(img, metric_type) for img, _ in images]
        
        # Get top-K indices
        largest = (ranking_mode == "Best (Highest scores)")
        top_indices = topk_indices(scores, k, largest=largest)
        
        # Create results dataframe
        results_data = []
        for rank, idx in enumerate(top_indices):
            results_data.append({
                "Rank": rank + 1,
                "Image": images[idx][1],
                "Score": scores[idx],
                "Index": idx
            })
        
        df = pd.DataFrame(results_data)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", len(images))
        with col2:
            st.metric("Top-K Selected", k)
        with col3:
            st.metric(f"{'Best' if largest else 'Worst'} Score", f"{df['Score'].iloc[0]:.4f}")
        with col4:
            st.metric("Average Score", f"{np.mean(scores):.4f}")
        
        # Display results table
        st.markdown(f"### 🎯 Top {k} {ranking_mode.split()[0]} Frames")
        st.dataframe(
            df[['Rank', 'Image', 'Score']],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization options
        st.markdown("### 🖼️ Visual Comparison")
        
        view_mode = st.radio(
            "View mode:",
            ["Grid View", "Detailed View", "Score Distribution"],
            horizontal=True
        )
        
        if view_mode == "Grid View":
            # Display in grid
            cols_per_row = st.slider("Columns per row:", 2, 5, 3)
            
            top_images = [images[idx][0] for idx in top_indices]
            top_labels = [f"#{rank+1}: {images[idx][1]}\nScore: {scores[idx]:.4f}" 
                         for rank, idx in enumerate(top_indices)]
            
            image_compare_grid(top_images, top_labels, columns=cols_per_row)
        
        elif view_mode == "Detailed View":
            # Detailed view with selection
            selected_rank = st.selectbox(
                "Select frame to view:",
                range(1, len(top_indices) + 1),
                format_func=lambda x: f"Rank #{x} - {df.loc[x-1, 'Image']} (Score: {df.loc[x-1, 'Score']:.4f})"
            )
            
            idx = top_indices[selected_rank - 1]
            img = images[idx][0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(img, caption=f"Rank #{selected_rank} - {images[idx][1]}", use_container_width=True)
            
            with col2:
                st.markdown("#### Frame Details")
                st.write(f"**File:** {images[idx][1]}")
                st.write(f"**Index:** {idx}")
                st.write(f"**Rank:** #{selected_rank}")
                st.write(f"**Score:** {scores[idx]:.6f}")
                
                # Additional metrics
                img_norm = img.astype(np.float32) / 255.0
                st.markdown("#### All Metrics")
                st.write(f"**Sharpness:** {variance_of_laplacian(img_norm):.4f}")
                st.write(f"**Gradient:** {mean_gradient_magnitude(img_norm):.4f}")
                gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
                st.write(f"**Brightness:** {mean_intensity(gray):.4f}")
                st.write(f"**Shape:** {img.shape}")
        
        else:  # Score Distribution
            st.markdown("#### 📈 Score Distribution")
            
            # Create histogram data
            score_df = pd.DataFrame({
                'Frame Index': range(len(scores)),
                'Score': scores,
                'Top-K': ['Selected' if i in top_indices else 'Not Selected' for i in range(len(scores))]
            })
            
            # Plot
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # All scores
            ax.hist(scores, bins=30, alpha=0.5, label='All Frames', color='blue', edgecolor='black')
            
            # Top-K scores
            top_scores = [scores[i] for i in top_indices]
            ax.hist(top_scores, bins=15, alpha=0.7, label=f'Top-{k} Frames', color='red', edgecolor='black')
            
            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Score Distribution - {metric_type}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Stats
            st.markdown("#### 📊 Distribution Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**All Frames:**")
                st.write(f"Mean: {np.mean(scores):.4f}")
                st.write(f"Std Dev: {np.std(scores):.4f}")
                st.write(f"Min: {np.min(scores):.4f}")
                st.write(f"Max: {np.max(scores):.4f}")
            
            with col2:
                st.markdown(f"**Top-{k} Frames:**")
                st.write(f"Mean: {np.mean(top_scores):.4f}")
                st.write(f"Std Dev: {np.std(top_scores):.4f}")
                st.write(f"Min: {np.min(top_scores):.4f}")
                st.write(f"Max: {np.max(top_scores):.4f}")
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Rankings to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"top{k}_{metric_type.replace(' ', '_').lower()}_rankings.csv",
                "text/csv"
            )
    
    else:
        st.info("👆 Select a data source from the sidebar to begin analysis")


if __name__ == "__main__":
    main()
