"""Real-time deblurring dashboard page using Streamlit."""

import streamlit as st
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.enhancement.deblur_net_torch import DeblurMobileNetV2UNet
from src.quality.blur_score import variance_of_laplacian, mean_gradient_magnitude
from src.quality.lowlight_score import mean_intensity


# Model configuration
MODEL_INPUT_SIZE = (256, 256)

# Default model paths (checked in order of preference)
DEFAULT_MODEL_PATHS = [
    Path("models/checkpoints/deblur_mobilenetv2_unet_torch_best.pt"),  # Standard training output
    Path("outputs/result/best_model.pt"),  # Alternative location
]


@st.cache_resource
def load_model(device: str):
    """Load and cache the deblur model.
    
    Searches for trained weights in default locations. Falls back to ImageNet-pretrained
    encoder if no trained weights found.
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in eval mode
    """
    model = DeblurMobileNetV2UNet(weights=None, backbone_trainable=False).to(device)
    
    # Search for trained weights
    weights_found = None
    for model_path in DEFAULT_MODEL_PATHS:
        if model_path.exists():
            weights_found = model_path
            break
    
    # Load weights if found
    if weights_found:
        try:
            state = torch.load(weights_found, map_location=device, weights_only=False)
            # Handle both checkpoint dict and raw state_dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            st.sidebar.success(f"✅ Model: {weights_found.name}")
        except Exception as e:
            st.sidebar.error(f"❌ Load failed: {str(e)[:50]}")
            st.sidebar.warning("⚠️ Using untrained model")
    else:
        st.sidebar.warning("⚠️ No trained model found")
        st.sidebar.info("🎓 Using ImageNet-pretrained encoder only")
        with st.sidebar.expander("📍 Searched locations"):
            for path in DEFAULT_MODEL_PATHS:
                st.code(str(path))
    
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess frame for model inference.
    
    Args:
        frame: Input frame (H, W, 3) BGR from OpenCV
    
    Returns:
        Preprocessed tensor (1, 3, H, W)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    frame_resized = cv2.resize(frame_rgb, MODEL_INPUT_SIZE)
    
    # Normalize to [0, 1] and convert to tensor
    frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
    
    # Rearrange HWC to CHW and add batch dimension
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor


def postprocess_frame(output_tensor: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    """Postprocess model output to displayable frame.
    
    Args:
        output_tensor: Model output tensor (1, 3, H, W) in [0, 1]
        original_size: Original frame size (H, W)
    
    Returns:
        RGB frame (H, W, 3) as uint8
    """
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu().detach()
    
    # Convert CHW to HWC
    output = output.permute(1, 2, 0).numpy()
    
    # Clip to [0, 1] and convert to uint8
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    
    # Resize back to original size
    output = cv2.resize(output, (original_size[1], original_size[0]))
    
    return output


def deblur_frame(model, frame: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    """Apply deblurring to a frame.
    
    Args:
        model: Deblur model
        frame: Input frame BGR from OpenCV
        device: Device to run inference on
    
    Returns:
        Tuple of (deblurred frame RGB, inference time)
    """
    start_time = time.time()
    original_size = frame.shape[:2]
    
    # Preprocess
    input_tensor = preprocess_frame(frame).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    deblurred = postprocess_frame(output, original_size)
    
    inference_time = time.time() - start_time
    
    return deblurred, inference_time


def calculate_frame_metrics(blurred_frame: np.ndarray, deblurred_frame: np.ndarray) -> dict:
    """Calculate comprehensive quality metrics comparing blurred and deblurred frames.
    
    Args:
        blurred_frame: Input blurred frame (RGB or BGR)
        deblurred_frame: Deblurred output frame (RGB)
    
    Returns:
        Dictionary with all metrics and improvements
    """
    # Convert blurred to RGB if needed
    if len(blurred_frame.shape) == 3 and blurred_frame.shape[2] == 3:
        if blurred_frame.dtype == np.uint8:
            blurred_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
        else:
            blurred_rgb = blurred_frame
    else:
        blurred_rgb = blurred_frame
    
    # Ensure both frames are same size
    if blurred_rgb.shape != deblurred_frame.shape:
        blurred_rgb = cv2.resize(blurred_rgb, (deblurred_frame.shape[1], deblurred_frame.shape[0]))
    
    # Normalize to [0, 1] float32
    if blurred_rgb.dtype == np.uint8:
        blurred_norm = blurred_rgb.astype(np.float32) / 255.0
    else:
        blurred_norm = blurred_rgb.astype(np.float32)
    
    if deblurred_frame.dtype == np.uint8:
        deblurred_norm = deblurred_frame.astype(np.float32) / 255.0
    else:
        deblurred_norm = deblurred_frame.astype(np.float32)
    
    # Convert to grayscale for some metrics
    blurred_gray = cv2.cvtColor((blurred_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    deblurred_gray = cv2.cvtColor((deblurred_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    
    # Blur scores (higher = sharper)
    blurred_laplacian = variance_of_laplacian(blurred_norm)
    deblurred_laplacian = variance_of_laplacian(deblurred_norm)
    
    blurred_gradient = mean_gradient_magnitude(blurred_norm)
    deblurred_gradient = mean_gradient_magnitude(deblurred_norm)
    
    # SSIM (higher = better, range [0, 1])
    ssim_score = ssim(blurred_gray, deblurred_gray, data_range=1.0)
    
    # PSNR (higher = better, typically 20-50 dB)
    psnr_score = psnr(blurred_norm, deblurred_norm, data_range=1.0)
    
    # MAE (lower = better, range [0, 1]) - calculated manually
    mae_score = np.mean(np.abs(blurred_norm - deblurred_norm))
    
    # Pixel Accuracy (percentage of pixels within threshold)
    pixel_diff = np.abs(blurred_norm - deblurred_norm)
    threshold = 0.1  # 10% difference threshold
    pixel_accuracy = np.mean(pixel_diff < threshold) * 100
    
    # Low light score (higher = brighter)
    blurred_brightness = mean_intensity(blurred_gray)
    deblurred_brightness = mean_intensity(deblurred_gray)
    
    # Calculate improvements
    def safe_improvement(new_val, old_val):
        if old_val == 0:
            return 0.0
        return ((new_val - old_val) / old_val) * 100
    
    laplacian_improvement = safe_improvement(deblurred_laplacian, blurred_laplacian)
    gradient_improvement = safe_improvement(deblurred_gradient, blurred_gradient)
    brightness_improvement = safe_improvement(deblurred_brightness, blurred_brightness)
    
    return {
        'blurred': {
            'laplacian_variance': blurred_laplacian,
            'gradient_magnitude': blurred_gradient,
            'brightness': blurred_brightness
        },
        'deblurred': {
            'laplacian_variance': deblurred_laplacian,
            'gradient_magnitude': deblurred_gradient,
            'brightness': deblurred_brightness
        },
        'comparison': {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'mae': mae_score,
            'pixel_accuracy': pixel_accuracy
        },
        'improvements': {
            'sharpness': laplacian_improvement,
            'gradient': gradient_improvement,
            'brightness': brightness_improvement
        }
    }


def main():
    st.title("🎬 Real-time Deblurring")
    
    # Initialize session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    if 'saved_frames' not in st.session_state:
        st.session_state.saved_frames = []
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")
    
    # Load model (searches default paths automatically)
    model = load_model(device)
    
    # Input source selection
    st.sidebar.markdown("## 📹 Input Source")
    input_source = st.sidebar.radio("Select input source:", ["Webcam", "Upload Video"])
    
    frames_data = []
    
    if input_source == "Webcam":
        st.markdown("### 📷 Webcam Input")
        
        # Duration control
        st.sidebar.markdown("#### ⏱️ Capture Duration")
        use_duration = st.sidebar.checkbox("Auto-stop after duration", value=False)
        duration_seconds = st.sidebar.slider(
            "Duration (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            disabled=not use_duration,
            help="Webcam will automatically stop after this duration"
        )
        
        # Webcam controls
        col_start, col_stop = st.sidebar.columns(2)
        if col_start.button("▶️ Start", disabled=st.session_state.webcam_running):
            st.session_state.webcam_running = True
            st.session_state.capture_start_time = time.time()
            st.rerun()
        if col_stop.button("⏹️ Stop", disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False
            st.rerun()
        
        if st.session_state.webcam_running:
            video_capture = cv2.VideoCapture(0)
            
            if not video_capture.isOpened():
                st.error("Cannot open webcam. Please check your camera connection.")
                st.session_state.webcam_running = False
                video_capture.release()
                return
            
            # Create placeholders
            col1, col2 = st.columns(2)
            blurred_placeholder = col1.empty()
            sharp_placeholder = col2.empty()
            
            metrics_placeholder = st.empty()
            fps_placeholder = st.empty()
            
            frame_count = 0
            fps_values = []
            start_time_global = time.time()
            
            # Snapshot controls
            snapshot_button = st.sidebar.button("📸 Save Current Frame")
            
            # Duration tracking
            duration_placeholder = st.empty()
            
            # Stop button inside loop
            stop_loop = st.button("🛑 Stop Capture")
            if stop_loop:
                st.session_state.webcam_running = False
            
            try:
                while st.session_state.webcam_running:
                    # Check duration limit
                    if use_duration:
                        elapsed = time.time() - st.session_state.capture_start_time
                        remaining = duration_seconds - elapsed
                        if elapsed >= duration_seconds:
                            st.session_state.webcam_running = False
                            duration_placeholder.success(f"⏱️ Auto-stopped after {duration_seconds} seconds")
                            break
                        else:
                            duration_placeholder.info(f"⏱️ Time remaining: {remaining:.1f}s / {duration_seconds}s")
                    
                    ret, frame = video_capture.read()
                    if not ret:
                        st.warning("Failed to capture frame")
                        break
                
                # Deblur the frame
                deblurred, inference_time = deblur_frame(model, frame, device)
                fps = 1 / inference_time if inference_time > 0 else 0
                fps_values.append(fps)
                
                # Calculate comprehensive metrics
                metrics = calculate_frame_metrics(frame, deblurred)
                
                # Store frame data
                frames_data.append({
                    "frame_num": frame_count,
                    "inference_time": inference_time * 1000,  # ms
                    "metrics": metrics
                })
                
                # Display frames
                blurred_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                         caption="Blurred Input", 
                                         width="stretch")
                sharp_placeholder.image(deblurred, 
                                       caption="Deblurred Output", 
                                       width="stretch")
                
                # Save snapshot if requested
                if snapshot_button:
                    st.session_state.saved_frames.append({
                        'frame_num': frame_count,
                        'blurred': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy(),
                        'deblurred': deblurred.copy(),
                        'metrics': metrics
                    })
                    st.sidebar.success(f"📸 Saved frame {frame_count}")
                
                # Display metrics
                capture_elapsed = time.time() - st.session_state.capture_start_time
                avg_fps = np.mean(fps_values[-30:]) if len(fps_values) > 0 else 0
                
                time_info = f"**Runtime:** {capture_elapsed:.1f}s"
                if use_duration:
                    progress_pct = (capture_elapsed / duration_seconds) * 100
                    time_info = f"**Progress:** {capture_elapsed:.1f}s / {duration_seconds}s ({progress_pct:.0f}%)"
                
                metrics_placeholder.markdown(f"""
                **Frame {frame_count}** | **Current FPS:** {fps:.1f} | **Avg FPS (30f):** {avg_fps:.1f} | **Inference:** {inference_time*1000:.1f}ms | {time_info}
                
                | Metric | Blurred | Deblurred | Improvement |
                |--------|---------|-----------|-------------|
                | **Blur Score (Laplacian)** | {metrics['blurred']['laplacian_variance']:.3f} | {metrics['deblurred']['laplacian_variance']:.3f} | **{metrics['improvements']['sharpness']:+.1f}%** |
                | **Gradient Magnitude** | {metrics['blurred']['gradient_magnitude']:.4f} | {metrics['deblurred']['gradient_magnitude']:.4f} | **{metrics['improvements']['gradient']:+.1f}%** |
                | **Brightness (Low Light)** | {metrics['blurred']['brightness']:.3f} | {metrics['deblurred']['brightness']:.3f} | **{metrics['improvements']['brightness']:+.1f}%** |
                | **SSIM** | - | {metrics['comparison']['ssim']:.4f} | - |
                | **PSNR (dB)** | - | {metrics['comparison']['psnr']:.2f} | - |
                | **MAE** | - | {metrics['comparison']['mae']:.4f} | ↓ lower is better |
                | **Pixel Accuracy** | - | {metrics['comparison']['pixel_accuracy']:.1f}% | - |
                """)
                
                frame_count += 1
                
                # Limit frame buffer
                if len(frames_data) > 100:
                    frames_data = frames_data[-100:]
                    fps_values = fps_values[-100:]
            
            except Exception as e:
                st.error(f"Error during capture: {str(e)}")
            finally:
                video_capture.release()
                st.session_state.webcam_running = False
            
            if frame_count > 0:
                st.success(f"✅ Processed {frame_count} frames. Average FPS: {np.mean(fps_values):.1f}")
            
            # Export metrics option
            if frames_data and st.button("💾 Export Metrics to CSV"):
                import pandas as pd
                df = pd.DataFrame([{
                    'frame': f['frame_num'],
                    'inference_ms': f['inference_time'],
                    'blurred_laplacian': f['metrics']['blurred']['laplacian_variance'],
                    'blurred_gradient': f['metrics']['blurred']['gradient_magnitude'],
                    'blurred_brightness': f['metrics']['blurred']['brightness'],
                    'deblurred_laplacian': f['metrics']['deblurred']['laplacian_variance'],
                    'deblurred_gradient': f['metrics']['deblurred']['gradient_magnitude'],
                    'deblurred_brightness': f['metrics']['deblurred']['brightness'],
                    'ssim': f['metrics']['comparison']['ssim'],
                    'psnr': f['metrics']['comparison']['psnr'],
                    'mae': f['metrics']['comparison']['mae'],
                    'pixel_accuracy': f['metrics']['comparison']['pixel_accuracy'],
                    'sharpness_improvement_%': f['metrics']['improvements']['sharpness'],
                    'gradient_improvement_%': f['metrics']['improvements']['gradient'],
                    'brightness_improvement_%': f['metrics']['improvements']['brightness']
                } for f in frames_data])
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    "realtime_metrics.csv",
                    "text/csv"
                )
            
            # Display saved frames
            if st.session_state.saved_frames:
                st.markdown(f"### 📸 Saved Snapshots ({len(st.session_state.saved_frames)})")
                for idx, snap in enumerate(st.session_state.saved_frames):
                    with st.expander(f"Snapshot {idx+1} - Frame {snap['frame_num']}"):
                        col1, col2 = st.columns(2)
                        col1.image(snap['blurred'], caption="Blurred", width="stretch")
                        col2.image(snap['deblurred'], caption="Deblurred", width="stretch")
                        m = snap['metrics']
                        st.markdown(f"""
                        **Blur Score:** {m['blurred']['laplacian_variance']:.3f} → {m['deblurred']['laplacian_variance']:.3f} ({m['improvements']['sharpness']:+.1f}%)  
                        **SSIM:** {m['comparison']['ssim']:.4f} | **PSNR:** {m['comparison']['psnr']:.2f} dB | **MAE:** {m['comparison']['mae']:.4f}
                        """)
                
                if st.button("🗑️ Clear Saved Frames"):
                    st.session_state.saved_frames = []
                    st.rerun()
    
    else:  # Upload Video
        st.markdown("### 📁 Upload Video")
        
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_video.mp4")
            temp_path.write_bytes(uploaded_file.read())
            
            video_capture = cv2.VideoCapture(str(temp_path))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Video loaded: {total_frames} frames")
            
            if st.button("Process Video"):
                progress_bar = st.progress(0)
                
                # Create placeholders
                col1, col2 = st.columns(2)
                blurred_placeholder = col1.empty()
                sharp_placeholder = col2.empty()
                
                metrics_placeholder = st.empty()
                
                frame_count = 0
                fps_values = []
                
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # Deblur the frame
                    deblurred, inference_time = deblur_frame(model, frame, device)
                    fps = 1 / inference_time if inference_time > 0 else 0
                    fps_values.append(fps)
                    
                    # Calculate comprehensive metrics
                    metrics = calculate_frame_metrics(frame, deblurred)
                    
                    # Store frame data
                    frames_data.append({
                        "frame_num": frame_count,
                        "inference_time": inference_time * 1000,  # ms
                        "metrics": metrics
                    })
                    
                    # Update display every 10 frames
                    if frame_count % 10 == 0:
                        blurred_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                                 caption=f"Frame {frame_count}/{total_frames}", 
                                                 width="stretch")
                        sharp_placeholder.image(deblurred, 
                                               caption=f"Deblurred {frame_count}/{total_frames}", 
                                               width="stretch")
                        
                        progress_bar.progress(frame_count / total_frames)
                    
                    frame_count += 1
                
                video_capture.release()
                temp_path.unlink()
                
                # Performance summary
                avg_fps = np.mean(fps_values)
                total_time = sum([f['inference_time'] for f in frames_data]) / 1000
                st.success(f"✅ Processed {frame_count} frames | Avg FPS: {avg_fps:.1f} | Total Time: {total_time:.1f}s")
                
                # Export metrics
                if st.button("💾 Export Metrics to CSV"):
                    import pandas as pd
                    df = pd.DataFrame([{
                        'frame': f['frame_num'],
                        'inference_ms': f['inference_time'],
                        'blurred_laplacian': f['metrics']['blurred']['laplacian_variance'],
                        'blurred_gradient': f['metrics']['blurred']['gradient_magnitude'],
                        'blurred_brightness': f['metrics']['blurred']['brightness'],
                        'deblurred_laplacian': f['metrics']['deblurred']['laplacian_variance'],
                        'deblurred_gradient': f['metrics']['deblurred']['gradient_magnitude'],
                        'deblurred_brightness': f['metrics']['deblurred']['brightness'],
                        'ssim': f['metrics']['comparison']['ssim'],
                        'psnr': f['metrics']['comparison']['psnr'],
                        'mae': f['metrics']['comparison']['mae'],
                        'pixel_accuracy': f['metrics']['comparison']['pixel_accuracy'],
                        'sharpness_improvement_%': f['metrics']['improvements']['sharpness'],
                        'gradient_improvement_%': f['metrics']['improvements']['gradient'],
                        'brightness_improvement_%': f['metrics']['improvements']['brightness']
                    } for f in frames_data])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv,
                        f"video_metrics_{frame_count}frames.csv",
                        "text/csv"
                    )
                
                # Show performance charts
                st.markdown("### 📊 Performance Analysis")
                
                chart_data = {
                    "Frame": [f["frame_num"] for f in frames_data],
                    "Inference Time (ms)": [f["inference_time"] for f in frames_data],
                    "Blurred Sharpness": [f["metrics"]['blurred']['laplacian_variance'] for f in frames_data],
                    "Deblurred Sharpness": [f["metrics"]['deblurred']['laplacian_variance'] for f in frames_data],
                    "SSIM": [f["metrics"]['comparison']['ssim'] for f in frames_data],
                    "PSNR": [f["metrics"]['comparison']['psnr'] for f in frames_data],
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Sharpness Comparison**")
                    st.line_chart({
                        "Blurred": chart_data["Blurred Sharpness"],
                        "Deblurred": chart_data["Deblurred Sharpness"],
                    }, width="stretch")
                
                with col2:
                    st.markdown("**Inference Time**")
                    st.line_chart({
                        "Inference Time (ms)": chart_data["Inference Time (ms)"]
                    }, width="stretch")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**SSIM Score**")
                    st.line_chart({
                        "SSIM": chart_data["SSIM"]
                    }, width="stretch")
                
                with col4:
                    st.markdown("**PSNR (dB)**")
                    st.line_chart({
                        "PSNR": chart_data["PSNR"]
                    }, width="stretch")


if __name__ == "__main__":
    main()
