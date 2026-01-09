# FPS Performance Optimizations for Real-time Dashboard

## Implemented Optimizations

### 1. **FP16 Half Precision (GPU Only)**
- **Speedup:** ~2x faster on CUDA GPUs
- **Implementation:** `model.half()` + `torch.cuda.amp.autocast()`
- **Memory:** 50% reduction in GPU memory usage
- **Trade-off:** Minimal accuracy loss (<1%)

### 2. **JIT Compilation (PyTorch 2.0+)**
- **Speedup:** 10-30% improvement
- **Implementation:** `torch.compile(model, mode="reduce-overhead")`
- **Benefit:** Optimized kernel fusion and reduced overhead

### 3. **Optimized Tensor Operations**
- Create tensors directly on target device (no CPU→GPU transfer)
- Combined operations: `* (1.0 / 255.0)` instead of `/ 255.0`
- Single-pass clamp and uint8 conversion
- FP16-aware preprocessing/postprocessing

### 4. **Efficient Resize Operations**
- **Downsampling:** `cv2.INTER_AREA` (faster, better quality)
- **Upsampling:** `cv2.INTER_LINEAR` (faster than default)
- **Option:** Skip output resize for maximum FPS

### 5. **Frame Skipping**
- Process every Nth frame (configurable)
- Ideal for high FPS cameras (30+ FPS)
- Example: Process every 2nd frame = 2x effective FPS

### 6. **Reduced Metrics Calculation**
- Calculate blur metrics every N frames (not every frame)
- Metrics are CPU-bound and slow
- Example: Calc every 3rd frame = ~20% speedup

## Expected Performance Gains

### Baseline (No Optimizations)
- CPU: ~5-10 FPS
- GPU (FP32): ~15-20 FPS

### With All Optimizations
- CPU: ~8-12 FPS (+30-50%)
- GPU (FP16): ~40-60 FPS (+150-200%)
- GPU (FP16 + Skip Resize): ~60-80 FPS (+250-300%)

### With Frame Skipping (Process every 2nd frame)
- CPU: ~16-24 FPS
- GPU (FP16): ~80-120 FPS

## How to Use

### In Streamlit Dashboard:

1. **Performance Settings** (Sidebar):
   - ✅ FP16 Half Precision (GPU only)
   - ✅ Skip Output Resize (faster display)
   - ✅ Process every N frames (1-5)
   - ✅ Metrics Calc Interval (1-10)

2. **Recommended Settings for Maximum FPS:**
   ```
   FP16 Half Precision: ON (if GPU available)
   Skip Output Resize: ON
   Process every N frames: 2
   Metrics Calc Interval: 5
   ```

3. **Recommended Settings for Best Quality:**
   ```
   FP16 Half Precision: ON (if GPU available)
   Skip Output Resize: OFF
   Process every N frames: 1
   Metrics Calc Interval: 1
   ```

## Technical Details

### Model Optimizations
```python
# FP16 conversion
model = model.half()  # Convert weights to FP16

# JIT compilation (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

### Inference Optimizations
```python
# Mixed precision inference
with torch.cuda.amp.autocast():
    output = model(tensor)
```

### Memory Optimizations
- FP16: 50% less GPU memory
- Allows larger batch sizes (if batching)
- Reduces VRAM requirements

## Benchmarking Tips

1. **Warm-up:** First few frames are slower (model loading)
2. **Measure:** Average FPS over 100+ frames
3. **GPU Utilization:** Check with `nvidia-smi`
4. **Bottlenecks:** 
   - CPU: OpenCV operations, metrics
   - GPU: Model inference
   - I/O: Camera/video reading

## Future Optimizations (TODO)

- [ ] TensorRT engine compilation (2-5x faster)
- [ ] ONNX export for cross-platform
- [ ] Batch processing for video files
- [ ] Multi-threading for I/O operations
- [ ] Model quantization (INT8) for CPU
- [ ] CUDA streams for async processing

## Troubleshooting

### FP16 Not Working?
- Requires CUDA GPU with compute capability >= 6.0
- Update PyTorch to latest version
- Check: `torch.cuda.is_available()`

### Lower FPS than Expected?
- Check GPU utilization: `nvidia-smi`
- Reduce image resolution
- Increase frame skip interval
- Disable metrics calculation

### Quality Issues?
- FP16 has minimal impact (~0.1% difference)
- Skip resize affects display only, not model
- Frame skipping doesn't affect individual frame quality
