# 🚀 Optimized Realtime rPPG Pipeline - Documentation

## 📋 Tổng Quan

Pipeline realtime tối ưu cho phát hiện nhịp tim (rPPG) từ video webcam với các nâng cấp:

✅ **Multi-threading** (4 workers: capture, preprocess, inference, display)  
✅ **GPU Batch Inference** (tối ưu hóa GPU, async processing)  
✅ **Dynamic Model Selection** (tự chọn model tốt nhất)  
✅ **Metrics Tracking** (latency, throughput, memory usage)  
✅ **Model Export** (TorchScript + state dict)  
✅ **Data Pipeline** (batch buffering, prefetch, normalization)  

---

## 🎯 Hiệu Suất

### Model Performance (CPU - 6 cores)

| Model | FPS | Latency | Size | Best For |
|-------|-----|---------|------|----------|
| **Mobile** | 24.47 | 40.87 ms | 9.12 MB | ⚡ Real-time |
| **Efficient** | 20.48 | 48.84 ms | 9.12 MB | ⚖️ Balanced |
| **Advanced** | 20.61 | 48.53 ms | 9.12 MB | 🎯 Accuracy |

**Batch Processing:**
- Batch 4: **42-45 FPS** (optimal)
- Batch 8: **40-46 FPS** (best sustained)

---

## 📁 File Structure

```
web_cam_optimized.py          # Main pipeline (4 workers)
test_performance.py           # Performance benchmarking
run_pipeline.py              # Quick start launcher
BENCHMARK_REPORT.md          # Performance report
ket_qua/
  ├── mobile_torchscript.pt
  ├── mobile_state_dict.pth
  ├── efficient_torchscript.pt
  ├── efficient_state_dict.pth
  ├── advanced_torchscript.pt
  ├── advanced_state_dict.pth
  └── benchmark_*.json
```

---

## 🚀 Cách Sử Dụng

### 1. Quick Start (Khuyên dùng)
```bash
python run_pipeline.py
```
Interactive menu để chọn model và chế độ.

### 2. Run với Model Cụ Thể
```bash
# Mobile model (nhanh nhất)
python web_cam_optimized.py --model mobile

# Efficient model (cân bằng)
python web_cam_optimized.py --model efficient

# Advanced model (chính xác)
python web_cam_optimized.py --model advanced

# Auto-select best model
python web_cam_optimized.py --auto-model
```

### 3. Benchmark Mode (Không hiển thị, đo FPS)
```bash
python web_cam_optimized.py --model mobile --no-display
```

### 4. Export Models
```bash
python web_cam_optimized.py --model mobile --export-only
```

### 5. Performance Benchmark
```bash
# Benchmark một model
python test_performance.py --model mobile

# Benchmark tất cả models
python test_performance.py --all-models

# Auto-select và benchmark
python test_performance.py --auto-select
```

---

## 🔧 Advanced Options

### Pipeline Arguments
```bash
python web_cam_optimized.py \
  --model advanced \           # advanced|mobile|efficient
  --window-size 24 \           # Temporal window (16-32)
  --no-display \               # No webcam display
  --auto-model \               # Auto-select best model
  --export-only                # Export only, no run
```

### Benchmark Arguments
```bash
python test_performance.py \
  --model advanced \           # Model to benchmark
  --window-size 24 \           # Window size
  --all-models \               # Benchmark all
  --auto-select                # Auto-select best
```

---

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN DISPLAY LOOP                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CAPTURE     │  │  PREPROCESS  │  │  INFERENCE   │  │
│  │  WORKER      │→→│  WORKER      │→→│  WORKER      │  │
│  │ (30 FPS)     │  │ (Face Detect)│  │ (GPU Batch)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│       ↓                  ↓                   ↓           │
│  capture_queue    preprocess_queue   inference_queue    │
│  (maxsize=10)     (maxsize=10)        (maxsize=5)       │
│                                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  DISPLAY WORKER                                   │  │
│  │  - Update BPM from inference results              │  │
│  │  - Track metrics (FPS, latency, memory)           │  │
│  │  - Show webcam + stats                            │  │
│  │  - Export results                                 │  │
│  └──────────────────────────────────────────────────┘  │
│                       ↑                                  │
│                  display_queue                           │
│                  (maxsize=5)                             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Metrics Tracked
- **Capture FPS:** Camera frame rate
- **Inference Latency:** Model processing time
- **Throughput:** Frames per second
- **GPU Memory:** GPU VRAM usage
- **CPU Usage:** CPU utilization %
- **BPM:** Heart rate estimation

---

## 🎯 Dynamic Model Selection

Pipeline tự động chọn model tốt nhất:

```python
if GPU available:
    → Advanced (VRAM > 2GB)
elif CPU cores >= 8:
    → Efficient
else:
    → Mobile (fast, lightweight)
```

### Kích hoạt Auto-selection
```bash
python web_cam_optimized.py --auto-model
```

---

## 💾 Model Export

Models được tự động export vào `ket_qua/`:

### TorchScript Format
```python
# Load và chạy
model = torch.jit.load('ket_qua/mobile_torchscript.pt')
output = model(input_tensor)
```

### State Dict Format
```python
# Load vào model
model = MobilePhysNet()
model.load_state_dict(torch.load('ket_qua/mobile_state_dict.pth'))
```

### Model Info
```json
{
  "model_type": "mobile",
  "input_shape": [1, 3, 24, 32, 32],
  "output_shape": [24],
  "model_size_mb": 9.12,
  "device": "cpu",
  "metrics": {...}
}
```

---

## 🔍 Metrics Example Output

```
============================================================
PIPELINE METRICS SUMMARY
============================================================
capture_fps                   : 30.0
preprocess_fps               : 30.0
inference_fps                : 24.47
throughput_frames_per_sec    : 22.1
gpu_memory_used_mb           : 0.0
cpu_percent                  : 45.2
total_frames_processed       : 650
total_inferences             : 27
============================================================
```

---

## 🎥 Realtime Display Features

- **Live Webcam Feed:** 800x650 resolution
- **BPM Display:** Real-time heart rate
- **Signal Quality:** "GOOD" / "WEAK" indicator
- **Signal Graph:** Real-time rPPG signal visualization
- **Performance Metrics:** FPS counter
- **Face Detection Box:** Bounding box visualization

### Controls
- **Q** or **Esc:** Quit application
- **Auto-refresh:** Every frame

---

## ⚙️ Performance Tuning

### Để tăng FPS:
1. ✅ Sử dụng **Mobile model** (24.47 FPS)
2. ✅ Giảm **window_size** (16 → 24 → 32)
3. ✅ Tăng **detection_skip** (skip face detection frames)
4. ✅ Dùng **batch processing** (--batch-size 4)

### Để tăng độ chính xác:
1. ✅ Sử dụng **Advanced model**
2. ✅ Tăng **window_size** (32)
3. ✅ Tăng **signal_buffer** size
4. ✅ Giảm **detection_skip** (more frequent detection)

### GPU Optimization:
1. ✅ Batch size 4-8 optimal
2. ✅ Pre-allocate tensors
3. ✅ Use torch.cuda.synchronize()
4. ✅ Monitor GPU memory

---

## 🐛 Troubleshooting

### Lỗi: "Cannot open webcam"
```bash
# Kiểm tra camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Lỗi: "Model not found"
```bash
# Export model
python web_cam_optimized.py --export-only
```

### FPS thấp
```bash
# Kiểm tra CPU/GPU
python test_performance.py --auto-select

# Chạy benchmark mode
python web_cam_optimized.py --auto-model --no-display
```

### Memory leak
```bash
# Monitor memory
python test_performance.py --model mobile
# Check GPU memory: nvidia-smi
```

---

## 📈 Benchmark Results

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for detailed performance metrics.

### Summary
- **Mobile:** 24.47 FPS, 40.87 ms latency, 9.12 MB
- **Efficient:** 20.48 FPS, 48.84 ms latency, 9.12 MB
- **Advanced:** 20.61 FPS, 48.53 ms latency, 9.12 MB

---

## 🎓 Architecture Details

### Multi-threading Benefits
- **Capture:** Stable FPS from camera (no frame drops)
- **Preprocess:** Non-blocking face detection
- **Inference:** Async batch processing
- **Display:** Real-time visualization + metrics

### Queue-based Communication
- Thread-safe queues (maxsize limits)
- Automatic frame dropping if queue full
- No busy-waiting, efficient CPU usage

### Batch Processing
- Accumulate N frames → Single batch
- Optimal batch size: 4 (42-45 FPS)
- Prefetch next batch while inferring current

---

## 📞 Support & Issues

Benchmark results saved to:
```
ket_qua/benchmark_model_YYYYMMDD_HHMMSS.json
```

For debugging:
```bash
# Verbose mode
python web_cam_optimized.py --model mobile 2>&1 | tee pipeline.log

# Performance trace
python test_performance.py --all-models 2>&1 | tee benchmark.log
```

---

**Version:** 1.0  
**Last Updated:** 2026-06-11  
**Status:** ✅ Production Ready
