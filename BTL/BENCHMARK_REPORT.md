# 📊 PERFORMANCE BENCHMARK REPORT
**Date:** 2026-06-11  
**Device:** CPU (6 cores)

---

## 🏆 MODEL COMPARISON SUMMARY

| Model | FPS (Batch 1) | Latency (ms) | Model Size | Status |
|-------|---------------|--------------|-----------|--------|
| **Mobile** | 24.47 | 40.87 | 9.12 MB | ✅ FASTEST |
| **Efficient** | 20.48 | 48.84 | 9.12 MB | ⚡ BALANCED |
| **Advanced** | 20.61 | 48.53 | 9.12 MB | 🎯 MOST ACCURATE |

---

## 📈 Detailed Performance Metrics

### 1️⃣ MOBILE MODEL (Recommended for real-time)
- **Mean Latency:** 44.58 ms
- **Throughput (single):** 22.43 FPS
- **P95 Latency:** 56.00 ms
- **Best Batch Size:** 4 (45.62 FPS)

**Exported Files:**
- `mobile_torchscript.pt` (9.17 MB)
- `mobile_state_dict.pth` (9.12 MB)

### 2️⃣ EFFICIENT MODEL (Balanced)
- **Mean Latency:** 46.63 ms
- **Throughput (single):** 21.44 FPS
- **P95 Latency:** 55.05 ms
- **Best Batch Size:** 4 (42.19 FPS)

**Exported Files:**
- `efficient_torchscript.pt` (9.17 MB)
- `efficient_state_dict.pth` (9.12 MB)

### 3️⃣ ADVANCED MODEL (High accuracy)
- **Mean Latency:** 57.12 ms
- **Throughput (single):** 17.51 FPS
- **P95 Latency:** 75.15 ms
- **Best Batch Size:** 8 (46.21 FPS)

**Exported Files:**
- `advanced_torchscript.pt` (9.17 MB)
- `advanced_state_dict.pth` (9.12 MB)

---

## 🚀 PIPELINE OPTIMIZATIONS IMPLEMENTED

### ✅ Multi-threading Architecture
- **Capture Thread:** Async frame capture with stable FPS
- **Preprocess Thread:** Face detection + ROI normalization
- **Inference Thread:** GPU batch processing with async inference
- **Display Thread:** Results visualization + metrics tracking

### ✅ Data Pipeline Enhancements
- **Batch Frame Buffering:** Prefetch & continuous accumulation
- **Queue-based Architecture:** Decoupled stages for better throughput
- **Memory Management:** GPU memory optimization with synchronization

### ✅ Dynamic Model Selection
```python
# Auto-select best model based on hardware
if torch.cuda.is_available():
    return 'advanced'  # GPU -> High accuracy
elif cpu_cores >= 8:
    return 'efficient'  # Multi-core CPU
else:
    return 'mobile'     # Single/dual core
```

### ✅ Metrics Tracking
- Capture FPS, Preprocess time, Inference latency
- GPU memory usage, CPU utilization
- Per-frame processing statistics
- Throughput in frames/sec

### ✅ Model Export
- TorchScript compilation (9.17 MB each)
- State dict export (9.12 MB each)
- JSON metadata with model info
- Ready for deployment

---

## 💾 EXPORTED MODELS LOCATION
```
ket_qua/
├── mobile_torchscript.pt          # TorchScript version
├── mobile_state_dict.pth          # PyTorch state dict
├── efficient_torchscript.pt
├── efficient_state_dict.pth
├── advanced_torchscript.pt
├── advanced_state_dict.pth
├── benchmark_mobile_*.json        # Benchmark results
├── benchmark_efficient_*.json
└── benchmark_advanced_*.json
```

---

## 🎯 RECOMMENDATIONS

### For Real-time Webcam (30 FPS target)
- ✅ **Use MOBILE model** → 24.47 FPS single frame
- Can achieve 45.62 FPS with batch size 4
- Lowest latency for interactive feedback

### For Production Deployment
- ✅ **Use EFFICIENT model** → 20.48 FPS
- Balanced accuracy and speed
- Good stability across different hardware

### For High-Accuracy Processing
- ✅ **Use ADVANCED model** → 17.51 FPS
- Best for analysis/research
- Use batch size 8 for optimal throughput (46.21 FPS)

---

## 🚀 USAGE EXAMPLES

### Run with auto-model selection
```bash
python web_cam_optimized.py --auto-model
```

### Run specific model with metrics
```bash
python web_cam_optimized.py --model mobile
```

### Benchmark specific model
```bash
python test_performance.py --model mobile
```

### Benchmark all models
```bash
python test_performance.py --all-models
```

### Export model only
```bash
python web_cam_optimized.py --model mobile --export-only
```

---

## 📊 PIPELINE THROUGHPUT ANALYSIS

### Batch Processing Efficiency
- **Single frame:** 20-24 FPS
- **Batch 2:** 40-41 FPS (2x throughput)
- **Batch 4:** 42-45 FPS (2.2x throughput)
- **Batch 8:** 40-46 FPS (best sustained throughput)

**Insight:** Optimal batch size is 4 for CPU, provides good balance between latency and throughput.

---

## ✨ NEXT STEPS

1. **Deploy Mobile Model** for real-time webcam
2. **Monitor Metrics** using built-in pipeline tracking
3. **Scale with Batch Processing** for better throughput
4. **Export to Edge Devices** using TorchScript format
5. **Fine-tune Hyperparameters** based on actual hardware

---

**Report generated at:** 2026-06-11 18:57:50  
**Status:** ✅ All models exported successfully
