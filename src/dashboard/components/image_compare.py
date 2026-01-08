"""Image comparison component for side-by-side frame comparison with interactive slider."""

import streamlit as st
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def image_compare(img1: np.ndarray, img2: np.ndarray, label1: str = "Before", label2: str = "After", 
                  width: int = 700, show_labels: bool = True, starting_position: int = 50):
    """Display two images side-by-side with an interactive slider for comparison.
    
    Args:
        img1: First image (numpy array, RGB)
        img2: Second image (numpy array, RGB)
        label1: Label for first image
        label2: Label for second image
        width: Width of the comparison viewer in pixels
        show_labels: Whether to show labels on images
        starting_position: Initial slider position (0-100)
    """
    # Convert numpy arrays to PIL Images
    if isinstance(img1, np.ndarray):
        if img1.dtype != np.uint8:
            img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        img1_pil = Image.fromarray(img1)
    else:
        img1_pil = img1
    
    if isinstance(img2, np.ndarray):
        if img2.dtype != np.uint8:
            img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
        img2_pil = Image.fromarray(img2)
    else:
        img2_pil = img2
    
    # Ensure both images are the same size
    if img1_pil.size != img2_pil.size:
        img2_pil = img2_pil.resize(img1_pil.size, Image.Resampling.LANCZOS)
    
    # Calculate dimensions
    aspect_ratio = img1_pil.height / img1_pil.width
    height = int(width * aspect_ratio)
    
    # Resize images for display
    img1_display = img1_pil.resize((width, height), Image.Resampling.LANCZOS)
    img2_display = img2_pil.resize((width, height), Image.Resampling.LANCZOS)
    
    # Convert to base64 for HTML embedding
    def img_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    img1_b64 = img_to_base64(img1_display)
    img2_b64 = img_to_base64(img2_display)
    
    # Create HTML/CSS for image comparison slider
    html_code = f"""
    <style>
        .img-comp-container {{
            position: relative;
            width: {width}px;
            height: {height}px;
            margin: 0 auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .img-comp-img {{
            position: absolute;
            width: {width}px;
            height: {height}px;
            object-fit: cover;
        }}
        
        .img-comp-img.img-comp-overlay {{
            clip-path: polygon(0 0, var(--position, 50%) 0, var(--position, 50%) 100%, 0 100%);
            transition: clip-path 0.1s ease;
        }}
        
        .img-comp-slider {{
            position: absolute;
            z-index: 10;
            cursor: ew-resize;
            width: 4px;
            height: 100%;
            background-color: white;
            left: var(--position, 50%);
            transform: translateX(-50%);
            box-shadow: 0 0 8px rgba(0,0,0,0.5);
        }}
        
        .img-comp-slider:before {{
            content: '';
            position: absolute;
            width: 40px;
            height: 40px;
            background-color: white;
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        
        .img-comp-slider:after {{
            content: '⟷';
            position: absolute;
            width: 40px;
            height: 40px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 20px;
            color: #333;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .img-label {{
            position: absolute;
            top: 10px;
            padding: 5px 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 14px;
            font-weight: bold;
            border-radius: 4px;
            z-index: 5;
        }}
        
        .img-label-left {{
            left: 10px;
        }}
        
        .img-label-right {{
            right: 10px;
        }}
    </style>
    
    <div class="img-comp-container" id="compareContainer">
        <img src="data:image/png;base64,{img2_b64}" class="img-comp-img" alt="{label2}">
        <img src="data:image/png;base64,{img1_b64}" class="img-comp-img img-comp-overlay" alt="{label1}">
        <div class="img-comp-slider"></div>
        {'<div class="img-label img-label-left">' + label1 + '</div>' if show_labels else ''}
        {'<div class="img-label img-label-right">' + label2 + '</div>' if show_labels else ''}
    </div>
    
    <script>
        (function() {{
            const container = document.getElementById('compareContainer');
            const slider = container.querySelector('.img-comp-slider');
            const overlay = container.querySelector('.img-comp-overlay');
            let isMouseDown = false;
            
            function setPosition(e) {{
                const rect = container.getBoundingClientRect();
                let x = e.clientX - rect.left;
                x = Math.max(0, Math.min(x, rect.width));
                const percentage = (x / rect.width) * 100;
                
                container.style.setProperty('--position', percentage + '%');
            }}
            
            // Initialize position
            container.style.setProperty('--position', '{starting_position}%');
            
            slider.addEventListener('mousedown', (e) => {{
                isMouseDown = true;
                e.preventDefault();
            }});
            
            container.addEventListener('mousemove', (e) => {{
                if (isMouseDown) {{
                    setPosition(e);
                }}
            }});
            
            document.addEventListener('mouseup', () => {{
                isMouseDown = false;
            }});
            
            container.addEventListener('click', (e) => {{
                setPosition(e);
            }});
            
            // Touch support
            slider.addEventListener('touchstart', (e) => {{
                isMouseDown = true;
                e.preventDefault();
            }});
            
            container.addEventListener('touchmove', (e) => {{
                if (isMouseDown && e.touches.length > 0) {{
                    const rect = container.getBoundingClientRect();
                    let x = e.touches[0].clientX - rect.left;
                    x = Math.max(0, Math.min(x, rect.width));
                    const percentage = (x / rect.width) * 100;
                    container.style.setProperty('--position', percentage + '%');
                }}
            }});
            
            document.addEventListener('touchend', () => {{
                isMouseDown = false;
            }});
        }})();
    </script>
    """
    
    st.components.v1.html(html_code, height=height + 20)


def image_compare_grid(images: list, labels: list = None, columns: int = 2):
    """Display multiple images in a grid for comparison.
    
    Args:
        images: List of images (numpy arrays, RGB)
        labels: List of labels for each image
        columns: Number of columns in the grid
    """
    if labels is None:
        labels = [f"Image {i+1}" for i in range(len(images))]
    
    # Create grid layout
    cols = st.columns(columns)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        col_idx = idx % columns
        
        with cols[col_idx]:
            # Convert numpy to uint8 if needed
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            st.image(img, caption=label, use_container_width=True)


def image_diff_heatmap(img1: np.ndarray, img2: np.ndarray, colormap: str = 'hot'):
    """Display difference heatmap between two images.
    
    Args:
        img1: First image (numpy array, RGB)
        img2: Second image (numpy array, RGB)
        colormap: Matplotlib colormap name for heatmap
    
    Returns:
        Difference heatmap as numpy array
    """
    import cv2
    
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalize to [0, 1]
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0
    
    # Calculate absolute difference
    diff = np.abs(img1 - img2)
    
    # Convert to grayscale if RGB
    if len(diff.shape) == 3:
        diff = np.mean(diff, axis=2)
    
    # Apply colormap
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    diff_colored = cmap(diff)[:, :, :3]  # Remove alpha channel
    diff_colored = (diff_colored * 255).astype(np.uint8)
    
    return diff_colored


def image_metrics_overlay(img: np.ndarray, metrics: dict, position: str = 'top-left'):
    """Overlay metrics text on an image.
    
    Args:
        img: Image (numpy array, RGB)
        metrics: Dictionary of metric names and values
        position: Position of overlay ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    
    Returns:
        Image with metrics overlay
    """
    import cv2
    
    # Create a copy
    img_copy = img.copy()
    if img_copy.dtype != np.uint8:
        img_copy = (np.clip(img_copy, 0, 1) * 255).astype(np.uint8)
    
    # Prepare text
    text_lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" 
                  for k, v in metrics.items()]
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 10
    line_height = 20
    
    # Calculate background rectangle size
    max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] 
                     for line in text_lines])
    rect_height = len(text_lines) * line_height + padding * 2
    rect_width = max_width + padding * 2
    
    # Determine position
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top-right':
        x = img_copy.shape[1] - rect_width - 10
        y = 10
    elif position == 'bottom-left':
        x = 10
        y = img_copy.shape[0] - rect_height - 10
    else:  # bottom-right
        x = img_copy.shape[1] - rect_width - 10
        y = img_copy.shape[0] - rect_height - 10
    
    # Draw semi-transparent background
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (x, y), (x + rect_width, y + rect_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0, img_copy)
    
    # Draw text
    for i, line in enumerate(text_lines):
        text_y = y + padding + (i + 1) * line_height - 5
        cv2.putText(img_copy, line, (x + padding, text_y), font, font_scale, 
                   (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img_copy
