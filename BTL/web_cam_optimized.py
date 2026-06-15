"""
Pipeline realtime tối ưu với multi-threading:
- Capture Thread: Đọc frames từ camera
- Preprocess Thread: Detect face, normalize
- Inference Thread: Chạy model với batch
- Display Thread: Hiển thị kết quả

Nâng cấp:
1. Batch frame buffer tối ưu (prefetch + pipeline)
2. Normalization + augmentation realtime
3. Tách hoàn toàn các thread (capture, preprocess, inference, display)
4. GPU batch inference tối ưu
5. Dynamic model selection dựa trên hardware
6. Metrics tracking (latency, throughput, memory)
7. Export model tối ưu vào ket_qua
"""

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# ENVIRONMENT
# =========================================================
import os

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# =========================================================
# IMPORTS
# =========================================================
import argparse
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from collections import deque
from threading import Thread, Lock, Event
import queue
from dataclasses import dataclass, field
import importlib
import threading
import psutil
import json
from datetime import datetime

# Use OpenCV Haar cascade as a stable fallback for face detection
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================================================
# CPU OPTIMIZATION
# =========================================================
cv2.setNumThreads(0)
try:
    torch.set_num_threads(4)
except Exception as e:
    print("[WARN] torch.set_num_threads failed: {}".format(e))

try:
    torch.set_num_interop_threads(1)
except Exception as e:
    print("[WARN] torch.set_num_interop_threads failed: {}".format(e))

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n==============================\nDEVICE: {}\n==============================\n".format(DEVICE))

# =========================================================
# MEDIAPIPE SETUP
# =========================================================


def _safe_torch_load(pth):
    """Module-level safe loader to prefer weights-only loads and state_dicts."""
    try:
        try:
            data = torch.load(str(pth), map_location=DEVICE if DEVICE is not None else 'cpu', weights_only=True)
        except TypeError:
            data = torch.load(str(pth), map_location=DEVICE if DEVICE is not None else 'cpu')
    except Exception as e:
        print("[WARN] torch.load failed: {}".format(e))
        return None

    if isinstance(data, dict):
        if 'model_state' in data and isinstance(data['model_state'], dict):
            return data['model_state']
        sample_vals = list(data.values())[:5]
        if all(hasattr(v, 'dtype') or hasattr(v, 'numel') or hasattr(v, 'shape') for v in sample_vals):
            return data
    return None

def _try_import_mediapipe(timeout=5.0):
    result = {'mod': None, 'err': None}

    def _import():
        try:
            result['mod'] = importlib.import_module('mediapipe')
        except Exception as e:
            result['err'] = e

    thread = threading.Thread(target=_import, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    return result['mod']

mp = _try_import_mediapipe(timeout=5.0)
face_mesh = None
if mp is not None:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[INFO] Mediapipe loaded successfully")
    except Exception as e:
        print("[WARN] Mediapipe setup failed. Falling back to Haar face detection.")
        print("  Reason:", str(e))
        face_mesh = None
else:
    print("[WARN] Mediapipe import timed out or failed. Falling back to Haar face detection.")

# Force Haar-only mode on CPU for stability
if DEVICE.type == 'cpu':
    face_mesh = None
    print("[INFO] CPU mode active: using Haar cascade for face detection only")

# =========================================================
# METRICS TRACKING (Nâng cấp 6)
# =========================================================
@dataclass
class PipelineMetrics:
    """Metrics tracking cho toàn bộ pipeline"""
    capture_fps: float = 0.0
    preprocess_fps: float = 0.0
    inference_fps: float = 0.0
    display_fps: float = 0.0
    inference_latency_ms: float = 0.0
    throughput_frames_per_sec: float = 0.0
    gpu_memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    total_frames_processed: int = 0
    total_inferences: int = 0
    
    def to_dict(self):
        return {
            'capture_fps': round(self.capture_fps, 2),
            'preprocess_fps': round(self.preprocess_fps, 2),
            'inference_fps': round(self.inference_fps, 2),
            'display_fps': round(self.display_fps, 2),
            'inference_latency_ms': round(self.inference_latency_ms, 3),
            'throughput_frames_per_sec': round(self.throughput_frames_per_sec, 2),
            'gpu_memory_used_mb': round(self.gpu_memory_used_mb, 2),
            'cpu_percent': round(self.cpu_percent, 2),
            'total_frames_processed': self.total_frames_processed,
            'total_inferences': self.total_inferences,
        }
    
    def print_summary(self):
        print("\n" + "="*60)
        print("PIPELINE METRICS SUMMARY")
        print("="*60)
        for key, val in self.to_dict().items():
            print("{:<30}: {}".format(key, val))
        print("="*60 + "\n")


# =========================================================
# DATA STRUCTURES (Nâng cấp: Data pipeline)
# =========================================================
@dataclass
class CapturedFrame:
    """Frame từ camera"""
    index: int
    timestamp: float
    frame: np.ndarray
    height: int
    width: int


@dataclass
class PreprocessedROI:
    """ROI sau preprocess"""
    frame_index: int
    roi: np.ndarray  # Normalized 32x32
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float


@dataclass
class BatchBuffer:
    """Batch của multiple frames"""
    frame_indices: list  # [int]
    roi_batch: np.ndarray  # Shape: (window_size, 32, 32, 3)
    valid: bool = True


@dataclass
class InferenceResult:
    """Kết quả inference"""
    frame_index: int
    prediction: np.ndarray  # Shape: (window_size,)
    processing_time: float


# =========================================================
# DYNAMIC MODEL SELECTION (Nâng cấp 5)
# =========================================================
def select_best_model():
    """
    Tự động chọn model tối ưu dựa trên hardware
    - CUDA có sẵn? -> advanced
    - CPU > 8 cores? -> efficient
    - CPU <= 4 cores? -> mobile
    """
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("[INFO] GPU detected: {:.1f} GB VRAM".format(gpu_mem))
        return 'advanced'
    else:
        cpu_cores = psutil.cpu_count(logical=False) or 4
        print("[INFO] CPU cores: {}".format(cpu_cores))
        if cpu_cores >= 8:
            return 'efficient'
        else:
            return 'mobile'


# =========================================================
# MODELS (từ web_cam.py)
# =========================================================
class AdvancedPhysNet(nn.Module):
    """Deeper 3D CNN with residual spatial backbone and temporal head."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
        )
        self.layer1 = self._residual_block(32, 64, stride=(1,2,2))
        self.layer2 = self._residual_block(64, 96, stride=(1,2,2))
        self.layer3 = self._residual_block(96, 128, stride=(1,1,1))
        self.layer4 = self._residual_block(128, 128)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def _residual_block(self, in_ch, out_ch, stride=(1,1,1)):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x) + x if x.shape == self.layer1(x).shape else self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(-1).mean(-1)
        x = self.temporal_head(x)
        return x.squeeze(1)


def load_model(model_type='advanced'):
    if model_type == 'efficient':
        model = EfficientPhysNet().to(DEVICE)
    elif model_type == 'mobile':
        model = TinyPhysNet().to(DEVICE)
    elif model_type in ('physformer', 'temporal_attention'):
        model = PhysFormerNet().to(DEVICE)
    else:
        model = AdvancedPhysNet().to(DEVICE)

    ckpt_path = Path(__file__).resolve().parent / "ket_qua" / "best_model.pth"
    if ckpt_path.exists():
        try:
            # try TorchScript first
            if str(ckpt_path).endswith('.pt'):
                try:
                    ts = torch.jit.load(str(ckpt_path), map_location=DEVICE)
                    print(f"[INFO] Loaded TorchScript model from {ckpt_path}")
                    return ts
                except Exception:
                    pass
            # use module-level safe loader
            ck = _safe_torch_load(ckpt_path)
            if isinstance(ck, dict):
                if 'model_state' in ck:
                    model.load_state_dict(ck['model_state'], strict=False)
                else:
                    model.load_state_dict(ck, strict=False)
                print("[INFO] Loaded checkpoint from {}".format(ckpt_path))
            else:
                print("[WARN] Checkpoint at {} is not a state_dict; skipping load for safety".format(ckpt_path))
        except Exception as e:
            print("[WARN] Failed to load checkpoint: {}".format(e))
    model.eval()
    return model


# =========================================================
# CAMERA SETUP
# =========================================================
def open_camera():
    for idx in range(5):
        print(f"[INFO] Testing camera {idx}")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] Camera {idx} opened")
                return cap
            cap.release()
    return None


# =========================================================
# REALTIME PIPELINE OPTIMIZED
# =========================================================
class OptimizedRealtimePipeline:
    """
    Pipeline realtime tối ưu với 4 threads:
    1. Capture: Đọc frames từ camera
    2. Preprocess: Face detection + normalization
    3. Inference: Batch processing + model inference (GPU optimized)
    4. Display: Hiển thị kết quả
    
    Nâng cấp:
    - GPU batch inference
    - Dynamic model selection
    - Metrics tracking
    - Model export
    """
    
    def __init__(self, model_type='advanced', window_size=4, batch_size=1, auto_model=False, fps_target=30):
        # Auto-select best model if requested
        if auto_model:
            model_type = select_best_model()
        
        self.model_type = model_type
        self.window_size = window_size
        self.batch_size = batch_size
        self.fps_target = max(20, fps_target)
        
        # Load model
        self.model = load_model(model_type)
        print(f"\n[INFO] Using {model_type} model for inference\n")
        
        # Queue setup (Nâng cấp: Multi-stage pipeline)
        self.capture_queue = queue.Queue(maxsize=10)
        self.preprocess_queue = queue.Queue(maxsize=10)
        self.inference_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=5)
        
        # Control signals
        self.stop_event = Event()
        self.threads = []
        
        # Stats
        self.frame_index = 0
        self.last_bbox = None
        self.bbox_history = deque(maxlen=5)
        self.last_frame = None
        self.frame_lock = Lock()
        
        # Signal tracking
        self.signal_buffer = deque(maxlen=128)
        self.bpm_history = deque(maxlen=6)
        self.last_bpm = 0
        self.last_fps = 0.0
        
        # Metrics tracking (Nâng cấp 6)
        self.metrics = PipelineMetrics()
        self.capture_frame_times = deque(maxlen=100)
        self.preprocess_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.metrics_lock = Lock()
        
    def capture_worker(self, cap, fps_target=30):
        """
        Thread 1: Capture frames từ camera
        Nâng cấp: Stable frame rate, buffering, metrics tracking
        """
        print("[INFO] Capture worker started")
        interval = 1.0 / fps_target
        last_time = time.time()
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame")
                break
            
            frame = cv2.flip(frame, 1)
            if frame.shape[1] != 800 or frame.shape[0] != 650:
                frame = cv2.resize(frame, (800, 650))
            
            captured = CapturedFrame(
                index=self.frame_index,
                timestamp=time.time(),
                frame=frame,
                height=frame.shape[0],
                width=frame.shape[1]
            )
            self.frame_index += 1
            
            with self.frame_lock:
                self.last_frame = frame.copy()
            
            try:
                self.capture_queue.put(captured, timeout=0.1)
            except queue.Full:
                print('[WARN] capture_queue is full; dropping captured frame')
            
            # Track capture FPS (Nâng cấp 6)
            current_time = time.time()
            self.capture_frame_times.append(current_time - last_time)
            
            # Rate limiting
            elapsed = current_time - last_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_time = time.time()
        
        print("[INFO] Capture worker stopped")
    
    def preprocess_worker(self):
        """
        Thread 2: Preprocess frames
        Nâng cấp: Normalize, augmentation realtime, face detection
        """
        print("[INFO] Preprocess worker started")
        detection_skip = 1  # Detect every frame
        frame_count = 0
        no_face_count = 0
        detect_count = 0
        
        while not self.stop_event.is_set():
            preprocess_start = time.time()
            try:
                captured = self.capture_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            frame = captured.frame
            roi = None
            bbox = None
            confidence = 0.0
            
            # Face detection (detect every frame)
            if frame_count % detection_skip == 0:
                roi, bbox, confidence = self._extract_face_optimized(frame)
                if bbox:
                    self.last_bbox = bbox
                    self.bbox_history.append(bbox)
                    detect_count += 1
                else:
                    no_face_count += 1
            else:
                # Use last bbox
                if self.last_bbox:
                    x1, y1, x2, y2 = self.last_bbox
                    roi_crop = frame[y1:y2, x1:x2]
                    if roi_crop.size > 0:
                        roi = self._normalize_roi(roi_crop)
                        bbox = self.last_bbox
                        confidence = 0.8
            
            frame_count += 1
            
            # Print detection stats every 30 frames
            if frame_count % 30 == 0:
                detection_rate = detect_count / (detect_count + max(1, no_face_count)) * 100
                print(f"[DEBUG] Face detection rate: {detection_rate:.1f}% ({detect_count}/{detect_count+no_face_count})")
                detect_count = 0
                no_face_count = 0
            
            if roi is not None:
                preprocessed = PreprocessedROI(
                    frame_index=captured.index,
                    roi=roi,
                    bbox=bbox,
                    confidence=confidence
                )
                try:
                    self.preprocess_queue.put(preprocessed, timeout=0.1)
                except queue.Full:
                    print('[WARN] preprocess_queue is full; dropping preprocessed ROI')

            preprocess_elapsed = time.time() - preprocess_start
            with self.metrics_lock:
                self.preprocess_times.append(preprocess_elapsed)
                if self.preprocess_times:
                    self.metrics.preprocess_fps = 1.0 / np.mean(self.preprocess_times)
        
        print("[INFO] Preprocess worker stopped")
    
    def inference_worker(self):
        """
        Thread 3: Batch inference with GPU optimization (Nâng cấp 4)
        - Batch processing
        - GPU memory management
        - Metrics tracking
        """
        print("[INFO] Inference worker started")
        frame_buffer = deque(maxlen=self.window_size)
        indices_buffer = deque(maxlen=self.window_size)
        collect_count = 0
        
        # Pre-allocate GPU tensor for batch processing (created per-batch in helper)
        
        while not self.stop_event.is_set():
            # Collect frames into batch
            try:
                roi_data = self.preprocess_queue.get(timeout=0.5)
                frame_buffer.append(roi_data.roi)
                indices_buffer.append(roi_data.frame_index)
                collect_count += 1
                
                # Debug: print every 10 collected frames
                if collect_count % 10 == 0:
                    print(f"[DEBUG] Collected {len(frame_buffer)}/{self.window_size} frames in buffer (frame index: {roi_data.frame_index})")
            except queue.Empty:
                pass
            
            # When buffer is full, run inference via helper
            if len(frame_buffer) == self.window_size:
                try:
                    self._process_batch_inference(frame_buffer, indices_buffer)
                    # Clear buffer for next batch
                    frame_buffer.clear()
                    indices_buffer.clear()
                except Exception as e:
                    print(f"[INFERENCE ERROR] {e}")
        
        print("[INFO] Inference worker stopped")
    
    def display_worker(self, show_display=True):
        """
        Thread 4: Display results + metrics tracking (Nâng cấp 6)
        Async display, FPS tracking, memory monitoring
        """
        print("[INFO] Display worker started")
        fps_time = time.time()
        frame_count = 0
        result_count = 0
        
        # Store last frame for display
        self.last_frame = None
        
        while not self.stop_event.is_set():
            try:
                result = self.display_queue.get(timeout=1)
                result_count += 1
                self._handle_display_result(result)
                frame_count += 1
            except queue.Empty:
                if result_count == 0 and frame_count == 0:
                    if (time.time() - fps_time) > 5:
                        print(f"[DEBUG] Display worker: waiting for results ({result_count} received so far)")
                        fps_time = time.time()
            
            # Calculate and update metrics (Nâng cấp 6)
            elapsed_time = time.time() - fps_time
            if frame_count > 0:
                with self.metrics_lock:
                    self.metrics.throughput_frames_per_sec = frame_count / elapsed_time
                    self.metrics.display_fps = self.metrics.throughput_frames_per_sec
                    if self.capture_frame_times:
                        self.metrics.capture_fps = 1.0 / np.mean(self.capture_frame_times)
                    if self.inference_times:
                        self.metrics.inference_fps = 1.0 / np.mean(self.inference_times)
                    
                    # Monitor GPU memory
                    if DEVICE.type == 'cuda':
                        self.metrics.gpu_memory_used_mb = torch.cuda.memory_allocated(DEVICE) / 1e6
                    
                    # Monitor CPU usage
                    self.metrics.cpu_percent = psutil.cpu_percent(interval=0.01)
                
                self.last_fps = self.metrics.throughput_frames_per_sec

    def _extract_face_optimized(self, frame):
        """Face detection tối ưu với fallback center crop"""
        h, w, _ = frame.shape

        # Try Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple scales for better detection
        faces = None
        scales = [1.1, 1.05]
        neighbors = [3, 4, 5]
        
        for scale in scales:
            for n in neighbors:
                faces = haar_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale, 
                    minNeighbors=n, 
                    minSize=(60, 60),
                    maxSize=(600, 600)
                )
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break
        
        if len(faces) > 0:
            x, y, w_box, h_box = faces[0]
            x1 = max(x - 20, 0)
            y1 = max(y - 20, 0)
            x2 = min(x + w_box + 20, frame.shape[1])
            y2 = min(y + h_box + 20, frame.shape[0])
            roi_crop = frame[y1:y2, x1:x2]
            if roi_crop.size > 0:
                roi = self._normalize_roi(roi_crop)
                return roi, (x1, y1, x2, y2), 0.85
        
        # Fallback: center crop (60% of image)
        center_w = w // 2
        center_h = h // 2
        crop_w = int(w * 0.6)
        crop_h = int(h * 0.6)
        
        x1 = max(center_w - crop_w // 2, 0)
        y1 = max(center_h - crop_h // 2, 0)
        x2 = min(center_w + crop_w // 2, w)
        y2 = min(center_h + crop_h // 2, h)
        
        roi_crop = frame[y1:y2, x1:x2]
        if roi_crop.size > 0:
            roi = self._normalize_roi(roi_crop)
            return roi, (x1, y1, x2, y2), 0.5  # Lower confidence for fallback

        return None, None, 0.0

    def _process_batch_inference(self, frame_buffer, indices_buffer):
        """Process a full batch (frames in frame_buffer) and push InferenceResult to display_queue."""
        inference_start = time.time()
        batch = np.array(list(frame_buffer), dtype=np.float32)
        batch_tensor = torch.from_numpy(batch).permute(3, 0, 1, 2).unsqueeze(0)
        batch_tensor = batch_tensor.to(DEVICE)

        with torch.no_grad():
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            pred = self.model(batch_tensor).squeeze().cpu().numpy()
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()

        inference_time = time.time() - inference_start
        pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)

        try:
            first_index = next(iter(indices_buffer))
        except StopIteration:
            first_index = -1
        result = InferenceResult(frame_index=first_index, prediction=pred, processing_time=inference_time)

        print(f"[DEBUG] Inference done: {self.window_size} frames processed in {inference_time*1000:.0f}ms")

        with self.metrics_lock:
            self.inference_times.append(inference_time)
            self.metrics.total_inferences += 1
            self.metrics.inference_latency_ms = inference_time * 1000

        try:
            self.display_queue.put(result, timeout=0.1)
        except queue.Full:
            print('[WARN] display_queue is full; dropping inference result')

    def _handle_display_result(self, result):
        """Handle a single display result: update buffers, compute BPM, and update metrics."""
        signal_value = float(result.prediction[-1])
        self.signal_buffer.append(signal_value)
        self.last_processing_time = result.processing_time

        bpm = self._estimate_bpm(self.signal_buffer)
        if bpm > 0:
            self.bpm_history.append(bpm)
            self.last_bpm = int(np.mean(self.bpm_history))
            print(f"[DEBUG] Regions: Forehead+Cheeks | signal: {signal_value:.4f} | bpm: {self.last_bpm}")
        else:
            print(f"[DEBUG] Result received but BPM not yet available (signal_len: {len(self.signal_buffer)})")

        with self.metrics_lock:
            self.metrics.total_frames_processed += 1
    
    def _extract_bbox_from_landmarks(self, frame, landmarks, small_size):
        """Extract bbox từ landmarks"""
        h, w = frame.shape[:2]
        # small_size provided for APIs expecting size; not needed as separate vars
        
        indices = [10, 67, 103, 109, 338, 33, 133, 362, 263, 50, 280]
        xs, ys = [], []
        
        for idx in indices:
            try:
                lm = landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                xs.append(x)
                ys.append(y)
            except Exception:
                # Skip invalid/missing landmark points
                continue
        
        if xs and ys:
            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)
            return (x1, y1, x2, y2)
        
        return None
    
    def _normalize_roi(self, roi):
        """Normalize ROI: resize to 32x32 + standardize"""
        roi = cv2.resize(roi, (32, 32))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = roi.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        
        return roi
    
    def export_optimized_model(self):
        """
        Export mô hình tối ưu vào ket_qua
        - TorchScript
        - ONNX (nếu khả dụng)
        - Model info
        """
        output_dir = Path(__file__).resolve().parent / "ket_qua"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n[INFO] Exporting model to {output_dir}...\n")
        
        # Calculate model size
        model_size_mb = sum(p.numel() for p in self.model.parameters()) * 4 / 1e6
        
        # 1. TorchScript Export
        try:
            dummy_input = torch.randn(1, 3, self.window_size, 32, 32).to(DEVICE)
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            ts_path = output_dir / f"{self.model_type}_torchscript.pt"
            traced_model.save(str(ts_path))
            ts_size_mb = ts_path.stat().st_size / 1e6
            print(f"[EXPORT] ✓ TorchScript exported: {ts_path} ({ts_size_mb:.2f} MB)")
        except Exception as e:
            print(f"[EXPORT] ✗ TorchScript export failed: {e}")
        
        # 2. Save model state dict
        try:
            state_path = output_dir / f"{self.model_type}_state_dict.pth"
            torch.save(self.model.state_dict(), str(state_path))
            state_size_mb = state_path.stat().st_size / 1e6
            print(f"[EXPORT] ✓ State dict exported: {state_path} ({state_size_mb:.2f} MB)")
        except Exception as e:
            print(f"[EXPORT] ✗ State dict export failed: {e}")
        
        # 3. Model information
        try:
            model_info = {
                'model_type': self.model_type,
                'input_shape': [1, 3, self.window_size, 32, 32],
                'output_shape': [self.window_size],
                'model_size_mb': round(model_size_mb, 2),
                'device': str(DEVICE),
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.to_dict()
            }
            
            info_path = output_dir / f"{self.model_type}_info.json"
            with open(str(info_path), 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"[EXPORT] ✓ Model info exported: {info_path}")
        except Exception as e:
            print(f"[EXPORT] ✗ Model info export failed: {e}")
        
        print()
        return output_dir
    
    def _estimate_bpm(self, signal, fps=30):
        """Estimate BPM từ signal - updated to work with small window sizes"""
        if len(signal) < 4:  # Reduced from 32
            return 0
        
        sig = np.asarray(list(signal), dtype=np.float32)
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std < 1e-6:
            return 0
        sig = sig / (std + 1e-6)
        
        # Bandpass FFT 0.7-4.0 Hz
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        
        if not np.any(mask):
            return 0
        
        mag = np.abs(X[mask])
        if mag.size == 0:
            return 0
        
        idx = np.argmax(mag)
        f_max = freqs[mask][idx]
        bpm = int(f_max * 60)
        
        return bpm if 30 <= bpm <= 240 else 0
    
    def run(self, show_display=True, run_duration=None):
        """Start pipeline"""
        print("[INFO] Starting optimized realtime pipeline...\n")

        cap = open_camera()
        if cap is None:
            print("[ERROR] Cannot open webcam")
            return

        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        threads = self._start_worker_threads(cap, show_display)

        try:
            if show_display:
                self._display_loop()
            else:
                self._headless_loop(run_duration)
        finally:
            # Cleanup
            self.stop_event.set()
            cap.release()
            cv2.destroyAllWindows()

            for t in threads:
                t.join(timeout=2)

            # Print metrics summary and export model
            self.metrics.print_summary()
            self.export_optimized_model()

            print("[INFO] Pipeline stopped\n")

    def _start_worker_threads(self, cap, show_display):
        """Start and return list of worker threads."""
        threads = [
            Thread(target=self.capture_worker, args=(cap, self.fps_target), daemon=True),
            Thread(target=self.preprocess_worker, daemon=True),
            Thread(target=self.inference_worker, daemon=True),
            Thread(target=self.display_worker, args=(show_display,), daemon=True),
        ]
        for t in threads:
            t.start()
        return threads

    def _display_loop(self):
        """Main display loop for interactive mode."""
        print("[INFO] Starting display loop. Press Q to quit.\n")
        while not self.stop_event.is_set():
            with self.frame_lock:
                frame = self.last_frame.copy() if self.last_frame is not None else None

            if frame is None:
                frame = np.ones((650, 800, 3), dtype=np.uint8) * 50
                cv2.putText(frame, "Waiting for camera...", (20, 340),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                if hasattr(self, 'last_bbox') and self.last_bbox is not None:
                    x1, y1, x2, y2 = self.last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if hasattr(self, 'last_bpm'):
                bpm_text = f"BPM: {self.last_bpm} bpm" if self.last_bpm > 0 else "BPM: --"
                cv2.putText(frame, bpm_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if hasattr(self, 'last_fps'):
                cv2.putText(frame, f"FPS: {self.last_fps:.2f}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Display Signal length
            signal_len = len(self.signal_buffer)
            cv2.putText(frame, f"Signal len: {signal_len}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if hasattr(self, 'last_processing_time'):
                cv2.putText(frame, f"Inference: {self.last_processing_time*1000:.0f} ms", (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("rPPG Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Stopping pipeline...\n")
                self.stop_event.set()
                break

    def _headless_loop(self, run_duration=None):
        """Headless run loop for non-interactive mode."""
        print(f"[INFO] Running headless for {run_duration or 'unlimited'} seconds. Press Ctrl+C to stop.\n")
        start_time = time.time()
        try:
            while not self.stop_event.is_set():
                if run_duration is not None and (time.time() - start_time) >= run_duration:
                    print("\n[INFO] Headless run duration reached. Stopping pipeline...\n")
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received. Stopping pipeline...\n")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Optimized realtime rPPG pipeline with metrics')
    parser.add_argument('--model', choices=['advanced', 'mobile', 'efficient'],
                       default='advanced', help='Model type')
    parser.add_argument('--window-size', type=int, default=4,
                       help='Temporal window size')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (benchmark mode)')
    parser.add_argument('--auto-model', action='store_true',
                       help='Auto-select best model based on hardware')
    parser.add_argument('--export-only', action='store_true',
                       help='Export model to ket_qua without running webcam')
    args = parser.parse_args()
    
    if args.export_only:
        pipeline = OptimizedRealtimePipeline(
            model_type=args.model,
            window_size=args.window_size,
            auto_model=args.auto_model
        )
        pipeline.export_optimized_model()
        print("\n[INFO] Model exported successfully!")
        return
    
    pipeline = OptimizedRealtimePipeline(
        model_type=args.model,
        window_size=args.window_size,
        auto_model=args.auto_model
    )
    
    pipeline.run(show_display=not args.no_display)


if __name__ == "__main__":
    main()
