import warnings
warnings.filterwarnings("ignore")

# =========================================================
# ENVIRONMENT
# =========================================================
import os

os.environ["GLOG_minloglevel"] = "3"

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

os.environ["TORCH_DISABLE_DYNAMO"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "0"

# Workaround for PyTorch Windows import hang during metadata registration
os.environ["TORCH_ALLOW_TF32"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# =========================================================
# IMPORTS
# =========================================================
import argparse
import cv2
import time
import numpy as np
import torch.nn as nn
from pathlib import Path
from collections import deque
from threading import Thread, Lock
import queue

# Lazy load torch to avoid Windows import hang
def _lazy_torch():
    global torch, DEVICE
    try:
        import torch as _torch
        torch = _torch
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print("[ERROR] Failed to import torch: {}".format(e))
        raise

# Placeholder torch object
torch = None
DEVICE = None

# Runtime optimization flags (can be toggled via CLI)
ENABLE_TORCH_COMPILE = False
ENABLE_DYNAMIC_QUANT = False
USE_PRUNED = False
ENABLE_FUSE_BN = False
ENABLE_PRUNE = False
PRUNE_AMOUNT = 0.3
ENABLE_TRACE = False
EXPORT_ONNX = False
BENCHMARK_LATENCY = False
BENCHMARK_RUNS = 50
ENABLE_STATIC_QUANT = False
ENABLE_QAT = False
QAT_EPOCHS = 0
DISTILL_TRAIN = False
DISTILL_EPOCHS = 0
DISTILL_ALPHA = 0.5
QUALITY_THRESHOLD = 0.6
RUNTIME_SELECTOR = False

# Load torch before using CPU tuning or device setup
_lazy_torch()

# Use OpenCV Haar cascade for a fast fallback face detector
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================================================
# CPU OPTIMIZATION
# =========================================================
cv2.setNumThreads(0)

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# =========================================================
# DEVICE
# =========================================================
print("\n==============================")
print("DEVICE: {}".format(DEVICE))
print("==============================\n")

# =========================================================
# MEDIAPIPE (guarded, timeout import to avoid sounddevice hang)
# =========================================================
import importlib
import threading


def _try_import_mediapipe(timeout=5.0):
    """Attempt to import mediapipe with a timeout to avoid blocking
    when optional audio dependencies (sounddevice/PortAudio) hang.
    Returns the module or None on failure/timeout.
    """
    result = {'mod': None, 'err': None}

    def _import():
        try:
            result['mod'] = importlib.import_module('mediapipe')
        except Exception as e:
            result['err'] = e

    t = threading.Thread(target=_import, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    if result['mod'] is not None:
        return result['mod']
    return None


mp = _try_import_mediapipe(timeout=5.0)
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
        print("[WARN] Mediapipe setup failed. Face detection disabled.")
        print("  Reason:", str(e))
        mp = None
        mp_face_mesh = None
        face_mesh = None
else:
    print("[WARN] Mediapipe import timed out or failed. Face detection disabled.")
    mp = None
    mp_face_mesh = None
    face_mesh = None

# =========================================================
# MTCNN (fallback face detector)
# =========================================================
try:
    from mtcnn import MTCNN
    mtcnn_detector = MTCNN()
    print("[INFO] MTCNN loaded as fallback detector")
except ImportError:
    mtcnn_detector = None
    print("[INFO] MTCNN not installed. Using MediaPipe only.")

# =========================================================
# MODEL
# =========================================================
class TinyPhysNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(16),

            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            ),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(32),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                (32, 1, 1)
            )
        )

        self.regressor = nn.Conv3d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        x = self.features(x)

        x = self.regressor(x)

        x = x.squeeze(1)
        x = x.squeeze(-1)
        x = x.squeeze(-1)

        return x


# =========================================================
# LARGER 3D CNN (PhysNet-style) for longer temporal windows
# =========================================================
class PhysNet3D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        self.head = nn.Conv3d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.head(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(1)
        return x


# =========================================================
# EFFICIENT 3D CNN + TRANSFORMER FOR rPPG
# =========================================================
class EfficientPhysNet(nn.Module):
    """Efficient 3D CNN backbone for rPPG with grouped convolutions and spatial pooling."""
    def __init__(self, in_channels=3, width=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
        )
        self.efficient_block1 = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1, groups=width, bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
            nn.Conv3d(width, width * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
        )
        self.efficient_block2 = nn.Sequential(
            nn.Conv3d(width * 2, width * 2, kernel_size=3, padding=1, groups=width * 2, bias=False),
            nn.BatchNorm3d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(width * 2, width * 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Conv1d(width * 4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.efficient_block1(x)
        x = self.efficient_block2(x)
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        return x.squeeze(1)


class PhysFormerNet(nn.Module):
    """Lightweight PhysFormer-style temporal transformer for rPPG."""
    def __init__(self, in_channels=3, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = x.mean(-1).mean(-1)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.head(x.permute(0, 2, 1))
        return x.squeeze(1)


class PhysFormerStub(PhysFormerNet):
    """Alias for backward compatibility with older scaffold names."""
    def __init__(self, in_channels=3, embed_dim=64, n_layers=4):
        super().__init__(in_channels=in_channels, embed_dim=embed_dim, num_layers=n_layers)


# =========================================================
# LOSS FUNCTIONS FOR rPPG
# =========================================================
def negative_pearson_loss(pred, target):
    """Negative Pearson correlation as loss (1 - corr)."""
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    pred_mean = pred - pred.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)
    
    num = (pred_mean * target_mean).sum(dim=1)
    den = torch.sqrt((pred_mean ** 2).sum(dim=1) * (target_mean ** 2).sum(dim=1) + 1e-8)
    corr = num / den
    loss = 1.0 - corr
    return loss.mean()


def mse_loss(pred, target):
    """Mean squared error loss for direct signal prediction."""
    return nn.MSELoss()(pred, target)


def correlation_loss(pred, target):
    """Correlation coefficient loss."""
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    pred_std = torch.std(pred_centered, dim=1, keepdim=True) + 1e-8
    target_std = torch.std(target_centered, dim=1, keepdim=True) + 1e-8
    
    pred_norm = pred_centered / pred_std
    target_norm = target_centered / target_std
    
    corr = (pred_norm * target_norm).mean(dim=1)
    return (1.0 - corr).mean()


def hybrid_loss(pred, target, weights=None, fs=30, low=0.7, high=4.0):
    """Hybrid loss combining Pearson, frequency, MSE, and correlation."""
    if weights is None:
        weights = {'pearson': 0.4, 'frequency': 0.2, 'mse': 0.2, 'correlation': 0.2}

    loss_p = negative_pearson_loss(pred, target)
    loss_f = frequency_loss(pred, target, fs=fs, low=low, high=high)
    loss_m = mse_loss(pred, target)
    loss_c = correlation_loss(pred, target)

    total = (weights.get('pearson', 0.4) * loss_p +
             weights.get('frequency', 0.2) * loss_f +
             weights.get('mse', 0.2) * loss_m +
             weights.get('correlation', 0.2) * loss_c)
    return total


def frequency_loss(pred, target, fs=30, low=0.7, high=4.0):
    """Loss on frequency domain in physiological band."""
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    n = pred.shape[1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(pred.device)
    p_spec = torch.abs(torch.fft.rfft(pred, dim=1))
    t_spec = torch.abs(torch.fft.rfft(target, dim=1))
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    p_band = p_spec[:, mask]
    t_band = t_spec[:, mask]
    p_norm = p_band / (p_band.sum(dim=1, keepdim=True) + 1e-8)
    t_norm = t_band / (t_band.sum(dim=1, keepdim=True) + 1e-8)
    return torch.mean((p_norm - t_norm) ** 2)
class TemporalAttentionPhysNet(nn.Module):
    """Temporal CNN with self-attention for improved temporal signal modeling."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
        )
        self.spatial_layers = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # Temporal attention: (B, C, T) -> attention weights
        self.temporal_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, bias=False)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)
        x = self.spatial_layers(x)
        x = x.mean(-1).mean(-1)  # (B, C, T)
        
        # Temporal self-attention
        x_t = x.permute(0, 2, 1)  # (B, T, C)
        attn_out, _ = self.temporal_attention(x_t, x_t, x_t)
        x_t = attn_out + x_t  # residual
        x = x_t.permute(0, 2, 1)  # (B, C, T)
        
        # Temporal prediction head
        x = self.temporal_head(x)
        return x.squeeze(1)


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1,1)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != (1,1,1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


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
        self.layer1 = Residual3DBlock(32, 64, stride=(1,2,2))
        self.layer2 = Residual3DBlock(64, 96, stride=(1,2,2))
        self.layer3 = Residual3DBlock(96, 128, stride=(1,1,1))
        self.layer4 = Residual3DBlock(128, 128)
        self.temporal_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(-1).mean(-1)
        x = self.temporal_head(x)
        return x.squeeze(1)


def instantiate_model(model_type='advanced'):
    if model_type == 'mobile':
        return MobilePhysNet()
    elif model_type == 'efficient':
        return EfficientPhysNet()
    elif model_type == 'physformer':
        return PhysFormerNet()
    elif model_type == 'temporal_attention':
        return TemporalAttentionPhysNet()
    elif model_type == 'rppg':
        # RPPG specialized model: 3D stem + TCN/Transformer temporal head + spatial attention
        return RPPGModel()
    return AdvancedPhysNet()


class TemporalConvNet(nn.Module):
    """Simple TCN stack for temporal modeling on sequence embeddings."""
    def __init__(self, in_channels, channels=(64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(prev, ch, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        return self.net(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for spatial attention (channel-wise)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        b, c, *_ = x.shape
        s = self.avg(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1, 1)
        return x * w


class RPPGModel(nn.Module):
    """Composite model for rPPG: 3D stem -> residual spatial -> temporal TCN -> Conv1d head.
    The forward returns a 2D tensor (B, T) representing the predicted rPPG signal.
    Additionally `last_confidence` attribute is set per forward call.
    """
    def __init__(self, in_channels=3, embed_dim=64, tcn_channels=(64, 64), window_frames=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
        )
        self.res = nn.Sequential(
            Residual3DBlock(32, 64, stride=(1,2,2)),
            Residual3DBlock(64, 96, stride=(1,2,2)),
        )
        self.se = SEBlock(96)
        self.project = nn.Conv3d(96, embed_dim, kernel_size=1)
        # temporal head expects (B, embed_dim, T)
        self.tcn = TemporalConvNet(embed_dim, channels=tcn_channels)
        self.head = nn.Conv1d(tcn_channels[-1], 1, kernel_size=1)
        self.conf_head = nn.Sequential(nn.Conv1d(tcn_channels[-1], 1, kernel_size=1), nn.Sigmoid())
        self.window_frames = window_frames
        self.last_confidence = None

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)
        x = self.res(x)
        x = self.se(x)
        x = self.project(x)
        x = x.mean(-1).mean(-1)  # (B, embed_dim, T)
        x = x.permute(0, 2, 1)  # (B, T, embed_dim)
        x = x.permute(0, 2, 1)  # back to (B, embed_dim, T) for Conv1d
        t = self.tcn(x)
        signal = self.head(t).squeeze(1)
        conf = self.conf_head(t).squeeze(1)
        # store confidence for external use
        try:
            # average confidence per sample
            self.last_confidence = conf.mean(dim=1).detach().cpu().numpy()
        except Exception:
            self.last_confidence = None
        return signal


def postprocess_pred_signal(pred_signal, fs=30, low=0.7, high=4.0):
    """Detrend, bandpass and estimate BPM from predicted signal array.
    Returns cleaned_signal, bpm
    """
    if pred_signal is None:
        return None, 0
    s = np.asarray(pred_signal).astype(np.float32)
    if s.size == 0:
        return s, 0
    # simple detrend: remove linear fit
    idx = np.arange(s.size)
    p = np.polyfit(idx, s, 1)
    trend = np.polyval(p, idx)
    s_d = s - trend

    # bandpass via FFT (reuse approach from estimate_bpm)
    n = s_d.size
    X = np.fft.rfft(s_d)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    Y = np.zeros_like(X)
    Y[mask] = X[mask]
    s_bp = np.fft.irfft(Y, n=n)

    bpm = estimate_bpm(s_bp, fps=fs)
    return s_bp, bpm


def _gather_sequential_fuse_pairs(model):
    """Gather module name pairs inside Sequential modules that are Conv->BatchNorm.
    Returns a list of lists suitable for torch.quantization.fuse_modules.
    """
    pairs = []
    try:
        for module_name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                sub_pairs = []
                for i in range(len(module) - 1):
                    a = module[i]
                    b = module[i + 1]
                    if (isinstance(a, (nn.Conv2d, nn.Conv3d)) and
                            isinstance(b, (nn.BatchNorm2d, nn.BatchNorm3d))):
                        sub_pairs.append(["{}.{}".format(module_name, i), "{}.{}".format(module_name, i+1)])
                pairs.extend(sub_pairs)
    except Exception as e:
        # If child inspection fails, return gathered pairs so far
        print('[DEBUG] _gather_sequential_fuse_pairs: {}'.format(e))
    return pairs


def fuse_conv_bn_in_model(model):
    """Attempt to fuse Conv+BatchNorm pairs using torch.quantization.fuse_modules.
    This is a best-effort helper that will skip if APIs are unavailable.
    """
    try:
        import torch.quantization as tq
        pairs = _gather_sequential_fuse_pairs(model)
        if pairs:
            try:
                tq.fuse_modules(model, pairs, inplace=True)
                print('[INFO] Fused Conv+BatchNorm pairs')
            except Exception as e:
                print('[WARN] fuse_modules failed: {}'.format(e))
    except Exception as e:
        print('[WARN] fuse_conv_bn_in_model: {}'.format(e))


def prune_model_weights(model, amount=0.3, structured=False):
    """Apply pruning to Conv and Linear modules. If structured=True, attempt structured pruning.
    This modifies modules in-place and leaves pruning reparametrization (masks) applied.
    After pruning for export, user should call `prune.remove(module, 'weight')` on modules.
    """
    try:
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                try:
                    if structured and hasattr(prune, 'ln_structured'):
                        prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                    else:
                        prune.l1_unstructured(module, name='weight', amount=amount)
                except Exception as e:
                    print('[WARN] prune failed for {}: {}'.format(name, e))
        print('[INFO] Applied pruning amount={} structured={}'.format(amount, structured))
    except Exception as e:
        print('[WARN] prune_model_weights: {}'.format(e))


def apply_dynamic_quantization(model):
    """Apply PyTorch dynamic quantization for supported layers on CPU.
    Returns quantized model or original on failure.
    """
    try:
        if torch is None:
            return model
        device_type = DEVICE.type if DEVICE is not None else 'cpu'
        if device_type != 'cpu':
            print('[WARN] Dynamic quantization is CPU-only; skipping (device != cpu)')
            return model
        q = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.Conv1d}, dtype=torch.qint8)
        print('[INFO] Applied dynamic quantization')
        return q
    except Exception as e:
        print('[WARN] apply_dynamic_quantization failed: {}'.format(e))
        return model


def compile_or_trace_model(model, example_input=None, do_compile=False, do_trace=False, export_path=None):
    """Optionally compile and/or trace the model. Returns possibly wrapped model.
    If do_trace and example_input provided, saves TorchScript to export_path if given.
    """
    m = model
    try:
        if do_compile and hasattr(torch, 'compile'):
            try:
                m = torch.compile(m)
                print('[INFO] torch.compile applied')
            except Exception as e:
                print('[WARN] torch.compile failed: {}'.format(e))

        if do_trace and example_input is not None:
            try:
                traced = torch.jit.trace(m, example_input)
                if export_path:
                    traced.save(str(export_path))
                    print('[INFO] Saved traced model to {}'.format(export_path))
                m = traced
            except Exception as e:
                print('[WARN] torch.jit.trace failed: {}'.format(e))
    except Exception as e:
        print('[WARN] compile_or_trace_model: {}'.format(e))
    return m


def benchmark_model_latency(model, input_shape=(1,3,32,32,32), device=torch.device('cpu'), runs=50, warmup=5):
    """Measure latency (ms) over multiple runs and print avg/p50/p95.
    Works with torch models; uses CPU or CUDA depending on device.
    """
    import statistics, time
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(dummy)
        times = []
        for _ in range(runs):
            t0 = time.time()
            _ = model(dummy)
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
    times.sort()
    avg = statistics.mean(times)
    p50 = times[int(0.50 * len(times))]
    p95 = times[int(0.95 * len(times)) if len(times) > 1 else -1]
    print('[BENCHMARK] runs={} avg={:.2f}ms p50={:.2f}ms p95={:.2f}ms'.format(runs, avg, p50, p95))
    return {'avg_ms': avg, 'p50_ms': p50, 'p95_ms': p95}


def remove_prune_masks(model):
    """Remove pruning reparametrization masks from modules before export.
    Best-effort: calls prune.remove(module, 'weight') when applicable.
    """
    try:
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                try:
                    prune.remove(module, 'weight')
                except Exception:
                    # not pruned or cannot remove
                    pass
        print('[INFO] Removed pruning masks where possible')
    except Exception as e:
        print('[WARN] remove_prune_masks: {}'.format(e))


def static_quantize_model(model, calibration_loader=None, backend='fbgemm'):
    """Apply static post-training quantization (PTQ) with optional calibration loader.
    This is best-effort and requires CPU execution for conversion.
    """
    try:
        if torch is None:
            return model
        torch.backends.quantized.engine = backend
        model.eval()
        fuse_conv_bn_in_model(model)
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)
        if calibration_loader is not None:
            # run calibration
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    inp = batch[0] if isinstance(batch, (list, tuple)) else batch
                    try:
                        model(inp)
                    except Exception:
                        pass
                    if i >= 10:
                        break
        torch.quantization.convert(model, inplace=True)
        print('[INFO] Static quantization applied (post-training)')
    except Exception as e:
        print('[WARN] static_quantize_model: {}'.format(e))
    return model


def prepare_qat(model):
    """Prepare model for Quantization Aware Training (QAT).
    Returns model prepared for training under QAT flows.
    """
    try:
        fuse_conv_bn_in_model(model)
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        print('[INFO] Model prepared for QAT')
    except Exception as e:
        print('[WARN] prepare_qat: {}'.format(e))
    return model


def finish_qat_and_convert(model):
    """Convert a QAT-trained model to quantized form.
    """
    try:
        model.eval()
        torch.quantization.convert(model, inplace=True)
        print('[INFO] QAT conversion complete')
    except Exception as e:
        print('[WARN] finish_qat_and_convert: {}'.format(e))
    return model


def distillation_train(student, teacher, dataloader, optimizer, epochs=3, alpha=0.5, temperature=2.0, device=None):
    """Knowledge distillation training loop (teacher frozen).
    Minimizes alpha*CE(student, labels) + (1-alpha)*KD_loss(student, teacher).
    """
    import torch.nn.functional as F
    if device is None:
        device = DEVICE if DEVICE is not None else torch.device('cpu')
    teacher.eval()
    student.train()
    for _ in range(epochs):
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                t_logits = teacher(inputs)

            s_logits = student(inputs)

            ce_loss = F.mse_loss(s_logits, targets)
            # distillation loss (soft targets)
            t_soft = F.log_softmax(t_logits / temperature, dim=1)
            s_soft = F.log_softmax(s_logits / temperature, dim=1)
            kd_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

            loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('[INFO] Distillation training complete')
    return student


def compute_signal_metrics(pred_signal, true_signal):
    """Compute simple SNR and MAE for rPPG-like signals.
    Returns dict with keys 'snr_db' and 'mae'.
    """
    try:
        pred = np.asarray(pred_signal, dtype=np.float32)
        true = np.asarray(true_signal, dtype=np.float32)
        mae = float(np.mean(np.abs(pred - true)))
        noise = pred - true
        signal_power = np.mean(true ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        snr = 10.0 * np.log10(signal_power / noise_power)
        return {'snr_db': float(snr), 'mae': float(mae)}
    except Exception as e:
        print('[WARN] compute_signal_metrics: {}'.format(e))
        return {'snr_db': 0.0, 'mae': float('inf')}


def runtime_selector(metrics, threshold=0.6):
    """Simple runtime selector: choose 'light' if metric (e.g., snr normalized) < threshold.
    This is a placeholder for a runtime quality-based selector.
    """
    try:
        score = metrics.get('snr_db', 0.0)
        # normalize rough SNR range to 0..1 (example heuristic)
        norm = min(max((score + 10) / 40.0, 0.0), 1.0)
        return 'heavy' if norm >= threshold else 'light'
    except Exception:
        return 'light'


def _try_load_torchscript(script_path):
    if not script_path.exists():
        return None
    try:
        model = torch.jit.load(str(script_path), map_location=DEVICE)
        model.eval()
        print("\n[INFO] Loaded TorchScript model from {}\n".format(script_path))
        return model
    except Exception as e:
        print("\n[WARN] Failed to load TorchScript model: {}\n".format(e))
        return None


def _safe_torch_load(pth):
    """Load a torch checkpoint safely: return state_dict-like dict or None."""
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


def _load_state_dict_safe(model, ck, ckpt_path):
    """Helper to load state_dict-like checkpoint into model; returns True if loaded."""
    try:
        if not isinstance(ck, dict):
            print("\n[WARN] Checkpoint at {} is not a state_dict; skipping load for safety\n".format(ckpt_path))
            return False
        if 'model_state' in ck and isinstance(ck['model_state'], dict):
            model.load_state_dict(ck['model_state'], strict=False)
        else:
            model.load_state_dict(ck, strict=False)
        print("\n[INFO] Loaded checkpoint from {}\n".format(ckpt_path))
        return True
    except Exception as e:
        print("\n[WARN] Failed applying state_dict from {}: {}\n".format(ckpt_path, e))
        return False


def _load_checkpoint_into_model(model, model_type):
    optimized_path = Path(__file__).resolve().parent / "ket_qua" / ("{}_best.pth".format(model_type))
    pruned_path = Path(__file__).resolve().parent / "ket_qua" / ("{}_pruned.pth".format(model_type))
    fallback_path = Path(__file__).resolve().parent / "ket_qua" / "best_model.pth"

    if USE_PRUNED and pruned_path.exists():
        ckpt_path = pruned_path
    else:
        ckpt_path = optimized_path if optimized_path.exists() else fallback_path

    if not ckpt_path.exists():
        print("\n[WARNING] No checkpoint found; using random init")
        return model

    try:
        # prefer TorchScript if provided
        if str(ckpt_path).endswith('.pt') or str(ckpt_path).endswith('.torch'):
            try:
                ts = torch.jit.load(str(ckpt_path), map_location=DEVICE)
                print("\n[INFO] Loaded TorchScript model from {}\n".format(ckpt_path))
                return ts
            except Exception:
                pass

        ck = _safe_torch_load(ckpt_path)
        # Try to apply checkpoint state dict using helper (prints warnings on failure)
        _load_state_dict_safe(model, ck, ckpt_path)
    except Exception as e:
        print("\n[WARN] Failed to load checkpoint {}: {}\n".format(ckpt_path, e))
    return model


def _apply_model_postprocessing(model, model_type):
    model.eval()
    model = _maybe_fuse_bn(model)
    model = _maybe_prune(model)
    model = _maybe_prepare_qat(model)
    model = _maybe_static_quant(model)
    model = _maybe_dynamic_quant(model)
    model = _maybe_compile_or_trace(model, model_type)
    _maybe_benchmark(model)
    _maybe_export_onnx(model, model_type)
    return model


def _maybe_fuse_bn(model):
    if not ENABLE_FUSE_BN:
        return model
    try:
        fuse_conv_bn_in_model(model)
    except Exception as e:
        print(f'[WARN] fuse_conv_bn_in_model failed: {e}')
    return model


def _maybe_prune(model):
    if not ENABLE_PRUNE:
        return model
    try:
        prune_model_weights(model, amount=PRUNE_AMOUNT, structured=False)
    except Exception as e:
        print(f'[WARN] prune_model_weights failed: {e}')
    return model


def _maybe_static_quant(model):
    if not ENABLE_STATIC_QUANT:
        return model
    try:
        # No calibration loader available here; run quick PTQ without calib if possible
        return static_quantize_model(model, calibration_loader=None)
    except Exception as e:
        print(f'[WARN] static_quantize_model failed: {e}')
        return model


def _maybe_prepare_qat(model):
    if not ENABLE_QAT:
        return model
    try:
        return prepare_qat(model)
    except Exception as e:
        print(f'[WARN] prepare_qat failed: {e}')
        return model


def _maybe_dynamic_quant(model):
    if not ENABLE_DYNAMIC_QUANT:
        return model
    try:
        return apply_dynamic_quantization(model)
    except Exception as e:
        print(f'[WARN] apply_dynamic_quantization failed: {e}')
        return model


def _maybe_compile_or_trace(model, model_type):
    try:
        example_input = None
        try:
            example_input = torch.randn(1, 3, 32, 32, 32).to(DEVICE if DEVICE is not None else torch.device('cpu'))
        except Exception:
            example_input = None

        return compile_or_trace_model(model, example_input=example_input,
                          do_compile=ENABLE_TORCH_COMPILE, do_trace=ENABLE_TRACE,
                          export_path=(Path(__file__).resolve().parent / 'ket_qua' / ("{}_traced.pt".format(model_type))) if ENABLE_TRACE else None)
    except Exception as e:
        print('[WARN] compile_or_trace_model failed: {}'.format(e))
        return model


def _maybe_benchmark(model):
    if not BENCHMARK_LATENCY:
        return
    try:
        device = torch.device('cpu') if DEVICE is None else DEVICE
        benchmark_model_latency(model, input_shape=(1,3,32,32,32), device=device, runs=BENCHMARK_RUNS)
    except Exception as e:
        print('[WARN] benchmark_model_latency failed: {}'.format(e))


def _maybe_export_onnx(model, model_type):
    if not EXPORT_ONNX:
        return
    try:
        export_path = Path(__file__).resolve().parent / 'ket_qua' / f"{model_type}.onnx"
        model_to_export = model
        if hasattr(model_to_export, 'eval'):
            model_to_export.eval()
        dummy = torch.randn(1,3,32,32,32).to(DEVICE if DEVICE is not None else torch.device('cpu'))
        torch.onnx.export(model_to_export, dummy, str(export_path), opset_version=13)
        print('[INFO] Exported ONNX to {}'.format(export_path))
    except Exception as e:
        print('[WARN] ONNX export failed: {}'.format(e))


def load_model(model_type='advanced'):
    script_path = Path(__file__).resolve().parent / "ket_qua" / ("{}_torchscript.pt".format(model_type))

    # Prefer TorchScript when available
    ts_model = _try_load_torchscript(script_path)
    if ts_model is not None:
        return ts_model

    model = instantiate_model(model_type).to(DEVICE)
    print("\n[INFO] Using {} model for inference\n".format(model_type))

    model = _load_checkpoint_into_model(model, model_type)

    model = _apply_model_postprocessing(model, model_type)

    return model


# =========================================================
# LIGHTWEIGHT MODEL + EXPORT / BENCHMARK HELPERS (Phase B)
# =========================================================
class MobilePhysNet(nn.Module):
    """Compact 3D CNN for low-latency CPU inference."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Conv1d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(-1).mean(-1)
        x = self.head(x)
        return x.squeeze(1)


def export_and_benchmark(model_type='advanced', export_path=None, runs=50):
    """Export model to TorchScript and benchmark CPU latency with dummy input."""
    device = torch.device('cpu')
    model = instantiate_model(model_type).to(device)

    # attempt to load pruned artifact into model if exists
    pruned_path = Path(__file__).resolve().parent / "ket_qua" / "advanced_pruned.pth"
    if pruned_path.exists():
        try:
            ck = _safe_torch_load(pruned_path)
            if isinstance(ck, dict):
                model.load_state_dict(ck, strict=False)
            else:
                print('[EXPORT] Pruned artifact not a state_dict; skipping')
            print('[EXPORT] Loaded pruned weights into export model')
        except Exception as e:
            print('[EXPORT] Failed to load pruned weights:', e)

    model.eval()
    dummy = torch.randn(1,3,32,32,32)
    # Trace
    try:
        traced = torch.jit.trace(model, dummy)
        if export_path:
            traced.save(str(export_path))
            print('[EXPORT] Saved TorchScript to {}'.format(export_path))
    except Exception as e:
        print('[EXPORT] TorchScript trace failed:', e)
        traced = None

    # Benchmark
    import time
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(dummy)
        t0 = time.time()
        for _ in range(runs):
            _ = model(dummy)
        t1 = time.time()
    avg_ms = (t1 - t0) / runs * 1000
    print('[BENCH] Avg latency: {:.2f} ms over {} runs'.format(avg_ms, runs))
    return avg_ms

# =========================================================
# AUTO CAMERA DETECTION
# =========================================================
def open_camera():

    for idx in range(5):

        print("[INFO] Testing camera {}".format(idx))

        cap = cv2.VideoCapture(
            idx,
            cv2.CAP_DSHOW
        )

        if cap.isOpened():

            print("[INFO] Camera {} opened".format(idx))

            return cap

        cap.release()

    return None

# =========================================================
# FACE ROI (UPGRADED WITH 5 IMPROVEMENTS)
# =========================================================
last_bbox = None
bbox_history = deque(maxlen=5)  # For ROI smoothing
last_orientation = None
last_landmarks = None  # Store current frame's landmarks for visualization
last_bpm = None
last_confidence_value = None
face_tracker = None
tracker_initialized = False

# Upgrade 5: Optimize MediaPipe - pre-configured lightweight model
if face_mesh is not None:
    face_mesh.static_image_mode = False
    face_mesh.max_num_faces = 1

def compute_3d_orientation(landmarks):
    """
    Upgrade 4: Extract 3D face orientation (yaw, pitch, roll) from landmarks.
    Uses key facial points to estimate head rotation.
    """
    try:
        # Key points for orientation estimation
        forehead = np.array([landmarks.landmark[10].x, landmarks.landmark[10].y, landmarks.landmark[10].z])
        chin = np.array([landmarks.landmark[152].x, landmarks.landmark[152].y, landmarks.landmark[152].z])
        
        left_ear = np.array([landmarks.landmark[234].x, landmarks.landmark[234].y, landmarks.landmark[234].z])
        right_ear = np.array([landmarks.landmark[454].x, landmarks.landmark[454].y, landmarks.landmark[454].z])
        
        # Compute vectors for orientation
        vertical = forehead - chin
        horizontal = right_ear - left_ear
        
        # Estimate angles
        pitch = np.arctan2(vertical[1], np.linalg.norm([vertical[0], vertical[2]])) * 180 / np.pi
        yaw = np.arctan2(horizontal[0], np.linalg.norm([horizontal[1], horizontal[2]])) * 180 / np.pi
        roll = 0  # Simplified; can be computed from eye symmetry
        
        return {"pitch": pitch, "yaw": yaw, "roll": roll}
    except Exception as e:
        print(f"[WARN] compute_3d_orientation: {e}")
        return None

def create_tracker():
    if hasattr(cv2, 'TrackerMOSSE_create'):
        return cv2.TrackerMOSSE_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
        return cv2.legacy.TrackerMOSSE_create()
    return None


def reset_tracker():
    global face_tracker, tracker_initialized
    face_tracker = None
    tracker_initialized = False


def init_face_tracker(frame, bbox):
    global face_tracker, tracker_initialized
    tracker = create_tracker()
    if tracker is None:
        return False

    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    try:
        tracker_initialized = tracker.init(frame, (x1, y1, w, h))
        face_tracker = tracker
        return tracker_initialized
    except Exception as e:
        print(f"[WARN] init_face_tracker failed: {e}")
        reset_tracker()
        return False


def update_tracked_bbox(frame):
    global face_tracker, tracker_initialized, last_bbox
    if face_tracker is None or not tracker_initialized:
        return None

    try:
        ok, bbox = face_tracker.update(frame)
    except Exception as e:
        print(f"[WARN] update_tracked_bbox update failed: {e}")
        reset_tracker()
        return None

    if not ok:
        reset_tracker()
        return None

    x, y, w, h = bbox
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), frame.shape[1])
    y2 = min(int(y + h), frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        reset_tracker()
        return None

    last_bbox = (x1, y1, x2, y2)
    return last_bbox


def extract_tracked_face(frame):
    bbox = update_tracked_bbox(frame)
    if bbox is None and last_bbox is None:
        return None

    if bbox is None:
        bbox = last_bbox

    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (32, 32))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (roi - mean) / std


def extract_multi_region_roi(frame, landmarks):
    """
    Upgrade 1: Use multiple landmark regions (forehead, eyes, cheeks, nose)
    instead of just forehead for better ROI coverage.
    """
    h, w, _ = frame.shape
    
    # Define landmark indices for different regions
    regions = {
        "forehead": [10, 67, 103, 109, 338],  # Original 5 points
        "eyes": [33, 133, 362, 263],  # Left & right eye corners
        "cheeks": [50, 280],  # Cheek points for better coverage
        "nose": [1, 4, 195],  # Nose points
    }
    
    all_xs, all_ys = [], []
    
    for region_name, indices in regions.items():
        for idx in indices:
            try:
                lm = landmarks.landmark[idx]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                all_xs.append(x)
                all_ys.append(y)
            except Exception as e:
                print(f"[WARN] extract_multi_region_roi: {e}")
    
    if not all_xs or not all_ys:
        return None
    
    x1 = max(min(all_xs) - 25, 0)
    y1 = max(min(all_ys) - 30, 0)
    x2 = min(max(all_xs) + 25, w)
    y2 = min(max(all_ys) + 30, h)
    
    return (x1, y1, x2, y2)

# =========================================================
# LANDMARK VISUALIZATION & TRACKING
# =========================================================
def get_landmark_region(landmark_idx):
    """
    Map landmark index to its facial region for color-coded visualization.
    Returns: (region_name, color_BGR)
    """
    # Define landmark groups by region (468 total landmarks)
    landmark_groups = {
        "lips": list(range(61, 69)) + list(range(78, 96)) + list(range(185, 205)) + list(range(409, 429)),
        "eyes": list(range(33, 48)) + list(range(133, 163)) + list(range(246, 259)) + list(range(362, 385)) + list(range(373, 386)),
        "eyebrows": list(range(65, 75)) + list(range(295, 303)) + list(range(51, 57)) + list(range(281, 287)),
        "nose": list(range(1, 6)) + list(range(195, 212)) + list(range(407, 424)),
        "face_contour": list(range(0, 17)) + list(range(17, 27)) + list(range(227, 232)) + list(range(447, 452)),
        "left_cheek": list(range(48, 65)) + list(range(209, 226)),
        "right_cheek": list(range(278, 295)) + list(range(426, 443)),
    }
    
    # Color scheme (BGR)
    color_map = {
        "lips": (0, 0, 255),          # Red
        "eyes": (255, 0, 0),          # Blue
        "eyebrows": (0, 165, 255),    # Orange
        "nose": (0, 255, 0),          # Green
        "face_contour": (255, 255, 0),  # Cyan
        "left_cheek": (255, 0, 255),  # Magenta
        "right_cheek": (128, 0, 128), # Purple
    }
    
    for region, indices in landmark_groups.items():
        if landmark_idx in indices:
            return region, color_map.get(region, (200, 200, 200))
    
    return "other", (200, 200, 200)

def _get_landmark_connections():
    """Get face mesh connection pairs (helper for draw_face_landmarks)."""
    return [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
        (10, 109), (109, 67), (67, 103), (103, 54), (54, 21),
        (46, 53), (53, 52), (52, 65), (65, 55), (55, 107),
        (276, 283), (283, 282), (282, 295), (295, 285), (285, 336),
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
        (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249),
    ]

def _draw_landmark_connections(frame, landmarks, h, w):
    """Draw connections between landmarks (helper for draw_face_landmarks)."""
    connections = _get_landmark_connections()
    for start_idx, end_idx in connections:
        try:
            start_lm = landmarks.landmark[start_idx]
            end_lm = landmarks.landmark[end_idx]
            if start_lm.z < 0 or end_lm.z < 0:
                continue
            start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
            end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
            _, color = get_landmark_region(start_idx)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1, cv2.LINE_AA)
        except Exception as e:
            print(f"[WARN] _draw_landmark_connections: {e}")

def _draw_landmark_points(frame, landmarks, h, w, landmark_size, confidence_threshold):
    """Draw landmark points (helper for draw_face_landmarks)."""
    for idx, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < confidence_threshold:
            continue
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        if not (0 <= x < w and 0 <= y < h):
            continue
        _, color = get_landmark_region(idx)
        cv2.circle(frame, (x, y), landmark_size, color, -1, cv2.LINE_AA)
        if landmark.visibility < 0.5:
            cv2.circle(frame, (x, y), landmark_size + 1, (100, 100, 100), 1, cv2.LINE_AA)

def draw_face_landmarks(frame, landmarks, draw_connections=True, landmark_size=2, confidence_threshold=0.1):
    """
    Draw MediaPipe Face Mesh landmarks on frame with region-based coloring.
    
    Args:
        frame: Input frame (BGR)
        landmarks: MediaPipe face landmarks object
        draw_connections: Whether to draw connections between landmarks
        landmark_size: Size of landmark circle
        confidence_threshold: Minimum confidence to draw landmark
    
    Returns:
        Frame with landmarks drawn
    """
    if landmarks is None:
        return frame
    
    h, w, _ = frame.shape
    
    if draw_connections:
        _draw_landmark_connections(frame, landmarks, h, w)
    
    _draw_landmark_points(frame, landmarks, h, w, landmark_size, confidence_threshold)
    
    return frame

def draw_face_bounding_box(frame, landmarks, bbox_color=(0, 255, 0), thickness=2):
    """
    Draw a tight bounding box around detected face landmarks.
    
    Args:
        frame: Input frame
        landmarks: MediaPipe face landmarks
        bbox_color: Color of bounding box (BGR)
        thickness: Line thickness
    
    Returns:
        Frame with bounding box drawn
    """
    if landmarks is None:
        return frame
    
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    xs = [lm.x * w for lm in landmarks.landmark if lm.visibility > 0.1]
    ys = [lm.y * h for lm in landmarks.landmark if lm.visibility > 0.1]
    
    if not xs or not ys:
        return frame
    
    x1 = max(int(min(xs)) - 10, 0)
    y1 = max(int(min(ys)) - 10, 0)
    x2 = min(int(max(xs)) + 10, w)
    y2 = min(int(max(ys)) + 10, h)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, thickness)
    
    return frame

def get_key_landmarks(landmarks):
    """
    Extract key facial landmarks for analysis.
    
    Returns:
        Dictionary with key landmark coordinates and 3D positions
    """
    if landmarks is None:
        return None
    
    try:
        key_indices = {
            "nose_tip": 4,
            "left_eye": 33,
            "right_eye": 263,
            "left_mouth": 61,
            "right_mouth": 291,
            "chin": 152,
            "forehead": 10,
        }
        
        key_landmarks = {}
        for name, idx in key_indices.items():
            lm = landmarks.landmark[idx]
            key_landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
        
        return key_landmarks
    except Exception as e:
        print(f"[WARN] get_key_landmarks failed: {e}")
        return None

def draw_key_landmarks_text(frame, key_landmarks, start_x=400, start_y=50):
    """
    Draw key landmark positions as text overlay.
    
    Args:
        frame: Input frame
        key_landmarks: Dictionary from get_key_landmarks()
        start_x, start_y: Starting position for text
    """
    if key_landmarks is None:
        return
    
    y_offset = start_y
    cv2.putText(frame, "Key Landmarks:", (start_x, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for name, coords in key_landmarks.items():
        y_offset += 20
        text = f"{name}: ({coords['x']:.3f}, {coords['y']:.3f}, {coords['z']:.3f})"
        cv2.putText(frame, text, (start_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def draw_landmark_statistics(frame, landmarks, start_x=10, start_y=350):
    """
    Draw landmark detection statistics.
    
    Args:
        frame: Input frame
        landmarks: MediaPipe face landmarks
        start_x, start_y: Starting position for text
    """
    if landmarks is None:
        return
    
    # Calculate statistics
    visibilities = [lm.visibility for lm in landmarks.landmark]
    avg_visibility = np.mean(visibilities) if visibilities else 0
    confident_count = sum(1 for v in visibilities if v > 0.5)
    
    stats_text = [
        f"Total Landmarks: {len(landmarks.landmark)}",
        f"Confident (>0.5): {confident_count}/{len(landmarks.landmark)}",
        f"Avg Visibility: {avg_visibility:.2f}",
    ]
    
    y_offset = start_y
    for text in stats_text:
        cv2.putText(frame, text, (start_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 25

def detect_fast_face_haar(frame):
    """Fast Haar cascade face ROI for mobile mode."""
    global last_bbox

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(24, 24),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
        else:
            return None
    else:
        x, y, w_box, h_box = faces[0]
        x1 = max(int(x) - 10, 0)
        y1 = max(int(y) - 10, 0)
        x2 = min(int(x + w_box) + 10, frame.shape[1])
        y2 = min(int(y + h_box) + 10, frame.shape[0])
        last_bbox = (x1, y1, x2, y2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (32, 32))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (roi - mean) / std


def _normalize_roi(roi):
    """Normalize ROI for neural network input (helper for extract_face)."""
    if roi is None or roi.size == 0:
        return None
    roi = cv2.resize(roi, (32, 32))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (roi - mean) / std

def _try_mediapipe_detection(rgb):
    """Try MediaPipe face mesh detection (helper for extract_face)."""
    global last_landmarks, last_orientation
    if face_mesh is None:
        return None
    try:
        results = face_mesh.process(rgb)
        if results is not None and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            last_landmarks = face_landmarks
            bbox = extract_multi_region_roi(rgb, face_landmarks)
            last_orientation = compute_3d_orientation(face_landmarks)
            return bbox
    except Exception as e:
        print(f"[WARN] _try_mediapipe_detection: {e}")
    return None

def _try_mtcnn_detection(frame, w, h):
    """Try MTCNN detection as fallback (helper for extract_face)."""
    if mtcnn_detector is None:
        return None
    try:
        dets = mtcnn_detector.detect_faces(frame)
        if len(dets) > 0:
            box = dets[0]['box']
            x1, y1, w_box, h_box = box
            x2 = x1 + w_box
            y2 = y1 + h_box
            return (max(int(x1) - 15, 0), max(int(y1) - 15, 0),
                    min(int(x2) + 15, w), min(int(y2) + 15, h))
    except Exception as e:
        print(f"[WARN] _try_mtcnn_detection: {e}")
    return None

def extract_face(frame, use_mobile=False):
    """
    Upgrade 2: Use Haar cascade as fast face detector, with MediaPipe/MTCNN fallback.
    Returns normalized ROI for neural network inference.
    """
    global last_bbox, last_landmarks

    if use_mobile:
        roi = detect_fast_face_haar(frame)
        if roi is not None and last_bbox is not None:
            init_face_tracker(frame, last_bbox)
        return roi

    roi = detect_fast_face_haar(frame)
    if roi is not None:
        init_face_tracker(frame, last_bbox)
        return roi

    h, w, _ = frame.shape
    current_bbox = None

    # Downsize for faster processing
    small = cv2.resize(frame, (160, 160))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Try MediaPipe next
    current_bbox = _try_mediapipe_detection(rgb)
    
    # Try MTCNN fallback if MediaPipe fails
    if current_bbox is None:
        current_bbox = _try_mtcnn_detection(frame, w, h)

    if current_bbox:
        last_bbox = current_bbox
        init_face_tracker(frame, current_bbox)
    elif last_bbox is None:
        return None

    if last_bbox is None:
        return None

    x1, y1, x2, y2 = last_bbox
    roi = frame[y1:y2, x1:x2]
    return _normalize_roi(roi)

# =========================================================
# BPM ESTIMATION (improved with bandpass FFT)
# =========================================================
def estimate_bpm(signal, fps=30):
    """Estimate BPM from 1D signal using bandpass FFT (0.7-4.0 Hz)."""
    if isinstance(signal, (list, tuple, deque)):
        sig = np.asarray(list(signal), dtype=np.float32)
    else:
        sig = np.asarray(signal, dtype=np.float32)

    # Allow shorter signals by zero-padding to 64 samples for earlier estimates
    min_len = 64
    if sig.size < min_len:
        sig = np.pad(sig, (0, min_len - sig.size), mode='constant', constant_values=0.0)

    sig = sig.flatten()
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-6)

    # Band-pass via FFT between 0.7 - 4.0 Hz
    def bandpass_fft(x, fs, low=0.7, high=4.0):
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        mask = (freqs >= low) & (freqs <= high)
        Y = np.zeros_like(X)
        Y[mask] = X[mask]
        y = np.fft.irfft(Y, n=len(x))
        return y, freqs[mask], np.abs(X[mask])

    _, freqs_masked, mag = bandpass_fft(sig, fps, low=0.7, high=4.0)
    if freqs_masked.size == 0 or mag.size == 0:
        return 0

    # dominant frequency in masked band
    idx = np.argmax(mag)
    f_max = freqs_masked[idx]
    bpm = int(f_max * 60)
    # sanity bounds
    if bpm < 30 or bpm > 240:
        return 0
    return bpm

# =========================================================
# DRAW SIGNAL
# =========================================================
def draw_signal(
    frame,
    signal
):

    if len(signal) < 2:

        return

    signal = np.array(
        signal,
        dtype=np.float32
    )

    signal = (
        signal - np.min(signal)
    ) / (
        np.max(signal)
        - np.min(signal)
        + 1e-8
    )

    h = 120
    w = 300

    start_x = 20
    start_y = 450

    pts = []

    for i in range(len(signal)):

        x = int(
            start_x
            + i * w / len(signal)
        )

        y = int(
            start_y
            - signal[i] * h
        )

        pts.append((x, y))

    for i in range(1, len(pts)):

        cv2.line(
            frame,
            pts[i - 1],
            pts[i],
            (0, 255, 0),
            2
        )

# =========================================================
# REALTIME WEBCAM (optimized with threading)
# =========================================================
# Thread-safe queue for frame buffers and results
class InferenceQueue:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True


class ModelContainer:
    """Holds a model reference that can be swapped at runtime in a thread-safe way."""
    def __init__(self, model):
        self.model = model
        self.lock = Lock()
        self.model_name = ''

def inference_worker(model_container, device, input_queue, output_queue):
    """Worker thread that processes frame buffers asynchronously."""
    while True:
        try:
            frame_buffer = input_queue.get(timeout=1)
            if frame_buffer is None:  # poison pill to exit
                break
            
            input_data = np.array(frame_buffer, dtype=np.float32)
            input_data = torch.from_numpy(input_data).permute(3, 0, 1, 2).unsqueeze(0)
            input_data = input_data.to(device)
            
            with torch.no_grad():
                # read current model (thread-safe)
                with model_container.lock:
                    current_model = model_container.model
                pred = current_model(input_data).squeeze().cpu().numpy()
                # try to read confidence if model exposes it
                conf_val = None
                try:
                    conf_attr = getattr(current_model, 'last_confidence', None)
                    if conf_attr is not None:
                        if isinstance(conf_attr, (list, tuple, np.ndarray)):
                            conf_val = float(conf_attr[0])
                        else:
                            conf_val = float(conf_attr)
                except Exception:
                    conf_val = None
            
            pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
            # put tuple(pred, confidence)
            output_queue.put((pred, conf_val))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[INFERENCE ERROR] {e}")
            continue

def _process_face_for_roi(frame, frame_index, detection_skip, use_mobile):
    """Process face detection and return ROI (helper for realtime_webcam)."""
    roi = None
    if frame_index % detection_skip == 0 or not tracker_initialized:
        roi = extract_face(frame, use_mobile=use_mobile)
    else:
        roi = extract_tracked_face(frame)
        if roi is None:
            roi = extract_face(frame, use_mobile=use_mobile)
    return roi

def _process_model_prediction(signal_buffer, bpm_history, output_q):
    """Extract and process model prediction from queue (helper for _update_signal_buffers)."""
    global last_bpm, last_confidence_value
    try:
        item = output_q.get_nowait()
        if isinstance(item, tuple) and len(item) == 2:
            pred, conf_val = item
        else:
            pred = item
            conf_val = None

        val = float(pred[-1])
        signal_buffer.append(val)
        bpm = estimate_bpm(signal_buffer, fps=30)
        last_bpm = bpm
        if conf_val is not None:
            try:
                last_confidence_value = float(conf_val)
            except Exception:
                last_confidence_value = None
        if bpm > 0:
            bpm_history.append(bpm)
        print(f"[RPPG-MODEL] bpm={bpm} signal_len={len(signal_buffer)} conf={last_confidence_value}")
        return True, bpm_history
    except queue.Empty:
        return False, bpm_history

def _process_fallback_signal(roi, signal_buffer, bpm_history):
    """Process fallback signal from ROI green channel (helper for _update_signal_buffers)."""
    global last_bpm
    try:
        if roi is not None:
            g_mean = float(np.mean(roi[..., 1]))
            signal_buffer.append(g_mean)
            bpm = estimate_bpm(signal_buffer, fps=30)
            last_bpm = bpm
            if bpm > 0:
                bpm_history.append(bpm)
            print(f"[RPPG-FALLBACK] bpm={bpm} signal_len={len(signal_buffer)} g_mean={g_mean:.6f}")
            return True, bpm_history
    except Exception as e:
        print(f"[WARN] _process_fallback_signal: {e}")
    return False, bpm_history

def _add_roi_to_buffer(roi, frame_buffer, input_q, window_size):
    """Add ROI to frame buffer and queue if full (helper for _update_signal_buffers)."""
    if roi is not None:
        frame_buffer.append(roi)
        if len(frame_buffer) == window_size:
            try:
                input_q.put_nowait(list(frame_buffer))
            except queue.Full:
                print('[WARN] input_q is full; dropping frame batch')

def _update_signal_buffers(roi, frame_buffer, signal_buffer, bpm_history, input_q, output_q, window_size):
    """Update signal and BPM buffers from output queue (helper for realtime_webcam)."""
    if roi is None and len(frame_buffer) > 0:
        roi = frame_buffer[-1]
    
    _add_roi_to_buffer(roi, frame_buffer, input_q, window_size)
    
    updated, bpm_history = _process_model_prediction(signal_buffer, bpm_history, output_q)
    if not updated:
        updated, bpm_history = _process_fallback_signal(roi, signal_buffer, bpm_history)
    
    return updated, bpm_history

def _render_rppg_overlay(frame, signal_buffer, bpm_history, overlay_mobile):
    """Render rPPG signal overlay on frame (helper for realtime_webcam)."""
    if overlay_mobile:
        return
    global last_bpm, last_confidence_value

    if len(bpm_history) > 0:
        smooth_bpm = int(np.mean(bpm_history))
        bpm_text = str(smooth_bpm)
    else:
        bpm_text = str(last_bpm) if (last_bpm is not None and last_bpm > 0) else "--"

    cv2.putText(frame, f"BPM: {bpm_text}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    signal_std = np.std(signal_buffer) if len(signal_buffer) > 0 else 0
    signal_quality = "✓ GOOD" if signal_std > 0.01 else "⚠ WEAK"
    cv2.putText(frame, f"Signal: {signal_quality}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if signal_std > 0.01 else (0, 165, 255), 1)
    draw_signal(frame, signal_buffer)
    # show model confidence if available
    if last_confidence_value is not None:
        try:
            conf_pct = int(last_confidence_value * 100)
            cv2.putText(frame, f"Conf: {conf_pct}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 1)
        except Exception:
            pass

def _render_face_overlay(frame, overlay_mobile):
    """Render face landmarks and orientation overlay (helper for realtime_webcam)."""
    if last_bbox is not None:
        x1, y1, x2, y2 = last_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if last_landmarks is not None and not overlay_mobile:
            frame = draw_face_landmarks(frame, last_landmarks, draw_connections=True, landmark_size=2)
            frame = draw_face_bounding_box(frame, last_landmarks, bbox_color=(0, 255, 0), thickness=2)
            key_lms = get_key_landmarks(last_landmarks)
            draw_key_landmarks_text(frame, key_lms, start_x=400, start_y=50)
            draw_landmark_statistics(frame, last_landmarks, start_x=10, start_y=350)

        if not overlay_mobile and last_orientation:
            yaw = last_orientation.get("yaw", 0)
            pitch = last_orientation.get("pitch", 0)
            roll = last_orientation.get("roll", 0)
            orientation_text = f"Yaw:{yaw:.0f}° Pitch:{pitch:.0f}° Roll:{roll:.0f}°"
            cv2.putText(frame, orientation_text, (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

def _setup_camera_properties(cap, fps_target, fast_mode):
    """Setup camera properties (helper for realtime_webcam)."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if fast_mode:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

def _setup_pipeline_params(model_type):
    """Setup detection and window parameters based on model type (helper for realtime_webcam)."""
    if model_type == 'mobile':
        return 15, 16, True  # detection_skip, window_size, overlay_mobile
    return 6, 24, False

def detect_cpu_capability():
    """Rudimentary CPU capability detection: returns 'low', 'medium', or 'high'."""
    try:
        cpu_count = os.cpu_count() or 1
        # try to get freq via psutil if available
        try:
            import psutil
            try:
                _freq_obj = psutil.cpu_freq()
                freq = getattr(_freq_obj, 'max', None)
            except Exception:
                freq = None
        except Exception:
            freq = None

        score = cpu_count
        if freq:
            score = cpu_count * (freq / 1000.0)

        if score < 4:
            return 'low'
        if score < 12:
            return 'medium'
        return 'high'
    except Exception:
        return 'medium'


def choose_initial_model_by_cpu(model_hint=None):
    cap = detect_cpu_capability()
    # If user explicitly passed a non-empty model hint (and it's not 'auto'), use it.
    if model_hint == 'auto':
        # fall through to CPU-based selection
        pass
    elif model_hint:
        return model_hint
    if cap == 'low':
        return 'mobile'
    if cap == 'medium':
        return 'efficient'
    return 'advanced'


def compute_realtime_quality(signal_buffer, confidence_value):
    """Compute a heuristic quality score (0..1) from signal std and confidence."""
    try:
        sig_std = float(np.std(signal_buffer)) if len(signal_buffer) > 0 else 0.0
        # normalize std: assume >0.1 is good, <0.01 is poor
        std_score = min(max((sig_std - 0.01) / 0.09, 0.0), 1.0)
        if confidence_value is None:
            conf_score = 0.5
        else:
            conf_score = min(max(float(confidence_value), 0.0), 1.0)
        return 0.5 * std_score + 0.5 * conf_score
    except Exception:
        return 0.5


def realtime_webcam(model_type='advanced', fast_mode=False, fps_target=30, auto_model=False):

    fps_target = max(30, min(300, fps_target))
    print(f"[INFO] Starting webcam with model={model_type}, fast_mode={fast_mode}, fps_target={fps_target}...\n")

    # Optionally auto-select model based on CPU capability
    # If auto_model is True, pass None so the chooser uses CPU detection; otherwise use provided model_type
    selected_model_type = choose_initial_model_by_cpu(None if auto_model else model_type)
    model = load_model(model_type=selected_model_type)
    cap = open_camera()

    if cap is None:
        print("[ERROR] Cannot open webcam")
        return

    _setup_camera_properties(cap, fps_target, fast_mode)
    detection_skip, window_size, overlay_mobile = _setup_pipeline_params(model_type)

    frame_buffer = deque(maxlen=window_size)
    signal_buffer = deque(maxlen=128)
    bpm_history = deque(maxlen=6)
    fps_time = time.time()
    
    inf_queue = InferenceQueue()
    input_q = inf_queue.frame_queue
    output_q = inf_queue.result_queue

    # model container for runtime swapping
    model_container = ModelContainer(model)
    model_container.model_name = selected_model_type
    worker_thread = Thread(target=inference_worker, args=(model_container, DEVICE, input_q, output_q), daemon=True)
    worker_thread.start()
    
    frame_index = 0
    global last_bbox, last_orientation, last_landmarks, tracker_initialized
    target_interval = 1.0 / fps_target
    previous_time = time.time()
    # Hand off to the extracted loop implementation
    # Prepare a loop context object to reduce parameter complexity
    loop_ctx = {
        'cap': cap,
        'frame_buffer': frame_buffer,
        'signal_buffer': signal_buffer,
        'bpm_history': bpm_history,
        'input_q': input_q,
        'output_q': output_q,
        'frame_index': frame_index,
        'detection_skip': detection_skip,
        'window_size': window_size,
        'overlay_mobile': overlay_mobile,
        'model_type': model_type,
        'fast_mode': fast_mode,
        'fps_time': fps_time,
        'previous_time': previous_time,
        'target_interval': target_interval,
        'worker_thread': worker_thread,
    }

    _run_webcam_loop(loop_ctx)

def _prepare_frame(frame):
    """Prepare frame: flip and resize (helper for realtime_webcam)."""
    frame = cv2.flip(frame, 1)
    if frame.shape[1] != 800 or frame.shape[0] != 650:
        frame = cv2.resize(frame, (800, 650))
    return frame

def _handle_display(frame, fast_mode):
    """Handle frame display and keyboard input (helper for realtime_webcam)."""
    if fast_mode:
        time.sleep(0.001)
        return False  # Continue loop
    
    cv2.putText(frame, "Press Q to quit", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Realtime rPPG", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        print("\n[INFO] Exiting...\n")
        return True  # Exit loop
    return False

def _enforce_fps_limit(previous_time, target_interval):
    """Enforce FPS limit and return new previous_time (helper for realtime_webcam)."""
    current_time = time.time()
    loop_elapsed = current_time - previous_time
    if loop_elapsed < target_interval:
        time.sleep(target_interval - loop_elapsed)
    return time.time()

def _run_webcam_loop(loop_ctx):
    """Run main webcam processing loop using a single context dict."""
    global last_bbox, last_orientation, last_landmarks, tracker_initialized

    # Unpack context
    cap = loop_ctx['cap']
    frame_buffer = loop_ctx['frame_buffer']
    signal_buffer = loop_ctx['signal_buffer']
    bpm_history = loop_ctx['bpm_history']
    input_q = loop_ctx['input_q']
    output_q = loop_ctx['output_q']
    frame_index = loop_ctx.get('frame_index', 0)
    detection_skip = loop_ctx['detection_skip']
    window_size = loop_ctx['window_size']
    overlay_mobile = loop_ctx['overlay_mobile']
    model_type = loop_ctx['model_type']
    fast_mode = loop_ctx['fast_mode']
    fps_time = loop_ctx['fps_time']
    previous_time = loop_ctx['previous_time']
    target_interval = loop_ctx['target_interval']
    worker_thread = loop_ctx['worker_thread']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _prepare_frame(frame)
        roi = _process_face_for_roi(frame, frame_index, detection_skip, model_type == 'mobile')
        frame_index += 1

        updated, bpm_history = _update_signal_buffers(roi, frame_buffer, signal_buffer, bpm_history,
                                                       input_q, output_q, window_size)

        # Adaptive selector: if runtime selector enabled and confidence low, swap to light model
        try:
            _maybe_swap_model_based_on_confidence(model_container, model_type, last_confidence_value)
        except Exception as e:
            print(f'[WARN] adaptive selector error: {e}')

        _render_face_overlay(frame, overlay_mobile)
        if updated:
            _render_rppg_overlay(frame, signal_buffer, bpm_history, overlay_mobile)

        fps = 1.0 / max((time.time() - fps_time), 1e-6)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if _handle_display(frame, fast_mode):
            break

        previous_time = _enforce_fps_limit(previous_time, target_interval)

    _cleanup_webcam(cap, worker_thread, input_q)

def _cleanup_webcam(cap, worker_thread, input_q):
    """Cleanup webcam resources (helper for realtime_webcam)."""
    try:
        input_q.put(None)
        worker_thread.join(timeout=2)
    except Exception as e:
        print(f"[WARN] _cleanup_webcam: {e}")
    cap.release()
    cv2.destroyAllWindows()


def _maybe_swap_model_based_on_confidence(model_container, model_type, last_confidence_value):
    """Helper to swap model based on confidence heuristic.

    If the runtime selector is disabled or confidence is unknown, this is a no-op.
    """
    if not RUNTIME_SELECTOR or last_confidence_value is None:
        return

    # simple threshold heuristic
    if last_confidence_value < QUALITY_THRESHOLD and model_type != 'mobile':
        # swap to mobile model for faster, more robust inference
        with model_container.lock:
            model_container.model = instantiate_model('mobile').to(DEVICE)
            print('[ADAPT] Swapped to mobile model due to low confidence')
    elif last_confidence_value >= QUALITY_THRESHOLD and model_type == 'mobile':
        # swap back to requested model
        with model_container.lock:
            model_container.model = instantiate_model(model_type).to(DEVICE)
            print('[ADAPT] Restored model based on confidence')

# =========================================================
# MAIN
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description='Realtime rPPG webcam runner')
    # Lightweight wrapper flags to call helper modules in this repo
    p.add_argument('--show-cpu', action='store_true', help='Print CPU info and chosen profile (adaptive_runtime)')
    p.add_argument('--assess-image', type=str, default=None, help='Assess image quality using signal_quality (path to image)')
    p.add_argument('--assess-signal', type=str, default=None, help='Assess 1D signal quality (npy file or comma list)')
    p.add_argument('--adaptive-demo', action='store_true', help='Run a small AdaptiveInferenceManager demo using FakeData')

    # Subcommands that forward to helper CLIs (forward remaining argv)
    subparsers = p.add_subparsers(dest='subcmd')
    p_export = subparsers.add_parser('export', help='Forward to export_models CLI')
    p_export.add_argument('args', nargs=argparse.REMAINDER)
    p_distill = subparsers.add_parser('distill', help='Forward to distill CLI')
    p_distill.add_argument('args', nargs=argparse.REMAINDER)

    # primary runtime flags
    p.add_argument('--model', choices=['advanced', 'mobile', 'efficient', 'physformer', 'temporal_attention'], default='advanced', help='Select runtime model type')
    p.add_argument('--benchmark', action='store_true', help='Run benchmark instead of webcam mode')
    p.add_argument('--export', action='store_true', help='Export selected model to TorchScript')
    p.add_argument('--fast', action='store_true', help='Run legacy webcam mode without display for faster processing')
    p.add_argument('--optimized', action='store_true', help='Run optimized pipeline (multi-threaded). Optimized is also the default mode.')
    p.add_argument('--legacy', action='store_true', help='Run legacy realtime webcam pipeline instead of optimized pipeline')
    p.add_argument('--auto-model', action='store_true', help='Auto-select best model (when using optimized pipeline)')
    p.add_argument('--no-display', action='store_true', help='Run optimized pipeline without display')
    p.add_argument('--fps', type=int, default=30, help='Target capture FPS for optimized or legacy pipeline (30-300)')
    p.add_argument('--run-duration', type=int, default=None, help='Run optimized pipeline headless for a fixed number of seconds')
    p.add_argument('--export-only', action='store_true', help='Export model from optimized pipeline and exit')

    # optimization & export flags
    p.add_argument('--torch-compile', action='store_true', help='Attempt to run torch.compile(model) when supported')
    p.add_argument('--quant-dynamic', action='store_true', help='Apply dynamic quantization (int8) on CPU when loading model')
    p.add_argument('--use-pruned', action='store_true', help='Prefer pruned checkpoint artifact when available')
    p.add_argument('--fuse-bn', action='store_true', help='Attempt to fuse Conv+BatchNorm where possible')
    p.add_argument('--prune', action='store_true', help='Apply pruning to model weights after loading')
    p.add_argument('--prune-amount', type=float, default=0.3, help='Amount (0-1) to prune when --prune is set')
    p.add_argument('--trace', action='store_true', help='Trace model with torch.jit.trace and save TorchScript')
    p.add_argument('--export-onnx', action='store_true', help='Export loaded model to ONNX')
    p.add_argument('--benchmark-latency', action='store_true', help='Run latency benchmark on loaded model')
    p.add_argument('--benchmark-runs', type=int, default=50, help='Number of runs for latency benchmark')
    p.add_argument('--static-quant', action='store_true', help='Apply static post-training quantization (PTQ)')
    p.add_argument('--qat', action='store_true', help='Prepare model for QAT (Quantization Aware Training)')
    p.add_argument('--qat-epochs', type=int, default=5, help='Epochs for QAT fine-tune (if enabled)')
    p.add_argument('--distill', action='store_true', help='Run knowledge distillation training (teacher->student)')
    p.add_argument('--distill-epochs', type=int, default=3, help='Epochs for distillation training')
    p.add_argument('--distill-alpha', type=float, default=0.5, help='Alpha weight for distillation loss')
    p.add_argument('--quality-threshold', type=float, default=0.6, help='Quality threshold for runtime selector (0-1)')
    p.add_argument('--runtime-selector', action='store_true', help='Enable runtime selector for light/heavy model switching')

    return p.parse_args()


def handle_wrappers_and_subcmds(args):
    """Handle lightweight wrappers and forward subcommands. Return True if handled and caller should exit."""
    # forward subcommands first
    if getattr(args, 'subcmd', None) == 'export':
        _forward_export(args)
        return True

    if getattr(args, 'subcmd', None) == 'distill':
        _forward_distill(args)
        return True

    # simple utility wrappers
    if args.show_cpu:
        _do_show_cpu()
        return True

    if args.assess_image:
        _do_assess_image(args.assess_image)
        return True

    if args.assess_signal:
        _do_assess_signal(args.assess_signal)
        return True

    if args.adaptive_demo:
        _do_adaptive_demo()
        return True

    return False


def _forward_export(args):
    try:
        import sys, export_models
        sys.argv = ['export_models.py'] + (args.args or [])
        export_models.main()
    except SystemExit:
        raise
    except Exception as e:
        print('Failed to run export_models:', e)


def _forward_distill(args):
    try:
        import sys, distill
        sys.argv = ['distill.py'] + (args.args or [])
        distill.main()
    except SystemExit:
        raise
    except Exception as e:
        print('Failed to run distill:', e)


def _do_show_cpu():
    try:
        from adaptive_runtime import get_cpu_info, choose_model_profile
        info = get_cpu_info()
        print('CPU info:', info)
        print('Chosen profile:', choose_model_profile(info))
    except Exception as e:
        print('Failed to get CPU info:', e)


def _do_assess_image(path):
    try:
        from signal_quality import assess_image_quality
        img = cv2.imread(path)
        score, conf = assess_image_quality(img)
        print(f'Image quality score={score:.3f} conf={conf:.3f}')
    except Exception as e:
        print('Failed to assess image:', e)


def _do_assess_signal(path):
    try:
        from signal_quality import assess_signal_quality
        if path.endswith('.npy'):
            s = np.load(path)
        else:
            s = np.array([float(x) for x in path.split(',')], dtype=np.float32)
        score, conf = assess_signal_quality(s)
        print(f'Signal quality score={score:.3f} conf={conf:.3f}')
    except Exception as e:
        print('Failed to assess signal:', e)


def _do_adaptive_demo():
    try:
        from adaptive_manager import AdaptiveInferenceManager
        import numpy as _np
        manager = AdaptiveInferenceManager({'light': 'models/light.pt', 'heavy': 'models/heavy.pt'})
        rng = _np.random.default_rng(123)
        dummy = rng.standard_normal(256)

        def light_fn(x):
            return {'profile': 'light', 'len': x.size}

        def heavy_fn(x):
            return {'profile': 'heavy', 'len': x.size}

        out = manager.run(dummy, light_fn, heavy_fn)
        print('Adaptive demo output:', out)
    except Exception as e:
        print('Adaptive demo failed:', e)


def configure_flags(args):
    global ENABLE_TORCH_COMPILE, ENABLE_DYNAMIC_QUANT, USE_PRUNED
    global ENABLE_FUSE_BN, ENABLE_PRUNE, PRUNE_AMOUNT, ENABLE_TRACE, EXPORT_ONNX, BENCHMARK_LATENCY, BENCHMARK_RUNS, ENABLE_STATIC_QUANT, ENABLE_QAT, QAT_EPOCHS, DISTILL_TRAIN, DISTILL_EPOCHS, DISTILL_ALPHA, QUALITY_THRESHOLD, RUNTIME_SELECTOR
    ENABLE_TORCH_COMPILE = bool(args.torch_compile)
    ENABLE_DYNAMIC_QUANT = bool(args.quant_dynamic)
    USE_PRUNED = bool(args.use_pruned)
    ENABLE_FUSE_BN = bool(args.fuse_bn)
    ENABLE_PRUNE = bool(args.prune)
    PRUNE_AMOUNT = float(args.prune_amount)
    ENABLE_TRACE = bool(args.trace)
    EXPORT_ONNX = bool(args.export_onnx)
    BENCHMARK_LATENCY = bool(args.benchmark_latency)
    BENCHMARK_RUNS = int(args.benchmark_runs)
    ENABLE_STATIC_QUANT = bool(args.static_quant)
    ENABLE_QAT = bool(args.qat)
    QAT_EPOCHS = int(args.qat_epochs)
    DISTILL_TRAIN = bool(args.distill)
    DISTILL_EPOCHS = int(args.distill_epochs)
    DISTILL_ALPHA = float(args.distill_alpha)
    QUALITY_THRESHOLD = float(args.quality_threshold)
    RUNTIME_SELECTOR = bool(args.runtime_selector)
    # heuristic: check tex.txt for adaptive inference hint
    try:
        tex_path = Path(__file__).resolve().parent / 'tex.txt'
        if tex_path.exists():
            try:
                content = tex_path.read_text(encoding='utf-8')
            except Exception:
                content = tex_path.read_text(encoding='utf-8', errors='ignore')
            if 'adaptive inference' in content.lower():
                RUNTIME_SELECTOR = True
    except Exception:
        pass
    if RUNTIME_SELECTOR:
        print(f'[INFO] Runtime selector enabled, threshold={QUALITY_THRESHOLD}')
    print(f"[INFO] Optimization flags: torch_compile={ENABLE_TORCH_COMPILE} dynamic_quant={ENABLE_DYNAMIC_QUANT} fuse_bn={ENABLE_FUSE_BN} prune={ENABLE_PRUNE} prune_amount={PRUNE_AMOUNT} trace={ENABLE_TRACE} export_onnx={EXPORT_ONNX} static_quant={ENABLE_STATIC_QUANT} qat={ENABLE_QAT} distill={DISTILL_TRAIN} runtime_selector={RUNTIME_SELECTOR}")


def run_pipeline(args):
    # Legacy benchmark/export using lightweight helpers
    if args.benchmark or args.export:
        export_path = Path(__file__).resolve().parent / 'ket_qua' / f'{args.model}_torchscript.pt'
        export_and_benchmark(model_type=args.model, export_path=export_path)
        return

    fps_target = max(30, min(300, args.fps))
    if fps_target != args.fps:
        print(f"[INFO] Clamping FPS to {fps_target} (valid range 30-300)")

    if args.legacy:
        realtime_webcam(model_type=args.model, fast_mode=args.fast, fps_target=fps_target)
        return

    # Optimized pipeline is now the default mode
    try:
        from web_cam_optimized import OptimizedRealtimePipeline
    except Exception as e:
        print(f"[ERROR] Optimized pipeline not available: {e}")
        print("Falling back to legacy realtime_webcam()")
        realtime_webcam(model_type=args.model, fast_mode=args.fast)
        return

    if args.export_only:
        pipeline = OptimizedRealtimePipeline(model_type=args.model, window_size=4, auto_model=args.auto_model)
        pipeline.export_optimized_model()
        print("\n[INFO] Optimized model exported to ket_qua/\n")
        return

    show_display = not args.no_display
    if args.fast:
        show_display = False

    pipeline = OptimizedRealtimePipeline(model_type=args.model, window_size=4, auto_model=args.auto_model, fps_target=fps_target)
    pipeline.run(show_display=show_display, run_duration=args.run_duration)


def main():
    args = parse_args()
    handled = handle_wrappers_and_subcmds(args)
    if handled:
        return
    configure_flags(args)
    run_pipeline(args)

# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":

    main()


