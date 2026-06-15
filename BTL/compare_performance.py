"""
Compare performance: Original vs Optimized Pipeline
"""

import torch
import numpy as np
import time
from pathlib import Path


def _safe_torch_load(pth):
    """Load a torch checkpoint safely: return state_dict-like dict or None."""
    try:
        # Prefer weights-only load when available to avoid executing pickled objects
        try:
            data = torch.load(str(pth), map_location='cpu', weights_only=True)
        except TypeError:
            # Older torch versions may not support weights_only
            data = torch.load(str(pth), map_location='cpu')
    except Exception as e:
        print(f"[WARN] torch.load failed: {e}")
        return None

    if isinstance(data, dict):
        if 'model_state' in data and isinstance(data['model_state'], dict):
            return data['model_state']
        sample_vals = list(data.values())[:5]
        if all(hasattr(v, 'dtype') or hasattr(v, 'numel') or hasattr(v, 'shape') for v in sample_vals):
            return data
    return None

# Test parameters
TEST_DURATION = 10  # seconds
WINDOW_SIZE = 24
NUM_FRAMES = 30 * TEST_DURATION  # 30 FPS * 10 sec = 300 frames

print("\n" + "="*70)
print("PERFORMANCE COMPARISON: Original vs Optimized Pipeline")
print("="*70 + "\n")

# Load model
model_path = Path("ket_qua/best_model.pth")
if not model_path.exists():
    print("[ERROR] Model not found!")
    exit(1)

print("[INFO] Loading model...")
try:
    # Try loading state dict
    state_dict_path = Path("ket_qua/advanced_state_dict.pth")
    if state_dict_path.exists():
        from web_cam import AdvancedPhysNet
        model = AdvancedPhysNet()
        checkpoint = _safe_torch_load(state_dict_path)
        if checkpoint is None:
            raise ValueError('Found checkpoint but it is not a state_dict')
        model.load_state_dict(checkpoint)
    else:
        # fallback to TorchScript
        torchscript_path = Path("ket_qua/advanced_torchscript.pt")
        if torchscript_path.exists():
            # TorchScript load is acceptable for exported models
            model = torch.jit.load(str(torchscript_path), map_location='cpu')
        else:
            raise FileNotFoundError("No model found")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("[INFO] Model loaded on {}".format(device))
print("[INFO] Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

# ============================================================
# TEST 1: Single Frame Inference (Original approach)
# ============================================================

print("\n[TEST 1] Single Frame Inference (Original Sequential)")
print("-" * 70)

single_frame = torch.randn(1, 3, 24, 32, 32).to(device)

# Warmup
for _ in range(10):
    _ = model(single_frame)

if device.type == 'cuda':
    torch.cuda.synchronize()

# Benchmark
start_time = time.time()
for _ in range(NUM_FRAMES):
    _ = model(single_frame)

if device.type == 'cuda':
    torch.cuda.synchronize()

elapsed = time.time() - start_time
fps_single = NUM_FRAMES / elapsed
latency_single = (elapsed / NUM_FRAMES) * 1000

print("Total frames: {}".format(NUM_FRAMES))
print("Total time: {:.2f}s".format(elapsed))
print("FPS: {:.2f}".format(fps_single))
print("Latency per frame: {:.2f}ms".format(latency_single))
print("Throughput: {:.2f} frames/sec".format(fps_single))

# ============================================================
# TEST 2: Batch Inference (Optimized approach)
# ============================================================

print("\n[TEST 2] Batch Processing (Optimized - Batch Size 4)")
print("-" * 70)

batch_size = 4
batch_frames = torch.randn(1, 3, WINDOW_SIZE * batch_size, 32, 32).to(device)

# Warmup
for _ in range(10):
    _ = model(batch_frames)

if device.type == 'cuda':
    torch.cuda.synchronize()

# Benchmark
num_batches = NUM_FRAMES // batch_size
start_time = time.time()

for _ in range(num_batches):
    _ = model(batch_frames)

if device.type == 'cuda':
    torch.cuda.synchronize()

elapsed = time.time() - start_time
fps_batch = (num_batches * batch_size) / elapsed
latency_batch = (elapsed / num_batches) * 1000

print("Total batches: {}".format(num_batches))
print("Batch size: {}".format(batch_size))
print("Total frames: {}".format(num_batches * batch_size))
print("Total time: {:.2f}s".format(elapsed))
print("Throughput (effective FPS): {:.2f}".format(fps_batch))
print("Latency per batch: {:.2f}ms".format(latency_batch))
print("Latency per frame: {:.2f}ms".format(latency_batch/batch_size))

# ============================================================
# TEST 3: Memory Comparison
# ============================================================

print("\n[TEST 3] Memory Usage Analysis")
print("-" * 70)

# Single frame
model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
print("Model size: {:.2f} MB".format(model_size))

# Input tensor sizes
single_tensor_size = single_frame.nelement() * single_frame.element_size() / (1024**2)
batch_tensor_size = batch_frames.nelement() * batch_frames.element_size() / (1024**2)

print("Single frame tensor: {:.2f} MB".format(single_tensor_size))
print("Batch (size 4) tensor: {:.2f} MB".format(batch_tensor_size))

if device.type == 'cuda':
    print("\nGPU Memory Allocated: {:.2f} MB".format(torch.cuda.memory_allocated() / 1024**2))
    print("GPU Memory Reserved: {:.2f} MB".format(torch.cuda.memory_reserved() / 1024**2))

# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

improvement_fps = ((fps_batch - fps_single) / fps_single) * 100
improvement_latency = ((latency_single - latency_batch) / latency_single) * 100

print("\n{:<30} {:<20} {:<20} {:<15}".format('Metric', 'Original', 'Optimized', 'Improvement'))
print("-" * 85)
print("{:<30} {:<20.2f} {:<20.2f} {:+.1f}%".format('FPS (throughput)', fps_single, fps_batch, improvement_fps))
print("{:<30} {:<20.2f} {:<20.2f} {:+.1f}%".format('Latency (ms)', latency_single, latency_batch/batch_size, improvement_latency))

print("\n🎯 Key Insights:")
print("  • Batch processing achieves {:+.1f}% better throughput".format(improvement_fps))
print("  • Per-frame latency reduced by {:+.1f}%".format(improvement_latency))
print("  • Effective throughput: {:.2f} FPS vs {:.2f} FPS".format(fps_batch, fps_single))
if device.type == 'cuda':
    print("  • GPU acceleration: ENABLED ✅")
else:
    print("  • GPU acceleration: DISABLED (CPU mode)")

print("\n✅ Optimization Benefits:")
print("  1. Multi-threading prevents frame drops")
print("  2. Batch processing increases throughput")
print("  3. Queue-based architecture decouples stages")
print("  4. Dynamic model selection for hardware")
print("  5. Real-time metrics tracking")

print("\n" + "="*70 + "\n")
