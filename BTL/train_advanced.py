"""
Minimal training pipeline for AdvancedPhysNet.
- Dataset loader looks for per-sample .npz files containing 'frames' (T,H,W,3) and 'ecg' (T,)
- If no real dataset found it creates a small synthetic dataset for smoke testing.
- Implements training loop, checkpointing, pruning/quantization placeholders.
"""
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from web_cam import AdvancedPhysNet, MobilePhysNet, EfficientPhysNet, PhysFormerNet, TemporalAttentionPhysNet, negative_pearson_loss, frequency_loss, mse_loss, correlation_loss, hybrid_loss

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Dataset"
CHECKPOINT_DIR = ROOT / "ket_qua"
CHECKPOINT_DIR.mkdir(exist_ok=True)

class RPpgDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, seq_len=32, transform=None):
        self.files = []
        self.seq_len = seq_len
        self.transform = transform
        # support .npz samples with 'frames' and 'ecg'
        for p in Path(data_dir).rglob("*.npz"):
            self.files.append(p)
        # If no files, create synthetic entries for smoke testing
        if len(self.files) == 0:
            self.synthetic = True
            self.length = 16
        else:
            self.synthetic = False
            self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.synthetic:
            # frames: (T, H, W, C)
            frames = np.random.rand(self.seq_len, 32, 32, 3).astype(np.float32)
            ecg = np.sin(np.linspace(0, 2 * np.pi * 1.2, self.seq_len)).astype(np.float32)
        else:
            p = self.files[idx]
            data = np.load(p)
            frames = data['frames']
            ecg = data['ecg']
            # ensure seq_len
            if frames.shape[0] >= self.seq_len:
                frames = frames[:self.seq_len]
                ecg = ecg[:self.seq_len]
            else:
                # pad
                pad_t = self.seq_len - frames.shape[0]
                frames = np.pad(frames, ((0,pad_t),(0,0),(0,0),(0,0)), mode='edge')
                ecg = np.pad(ecg, (0,pad_t), mode='edge')

        # convert to (C, T, H, W)
        frames = frames.transpose(3,0,1,2).copy()
        # normalize to 0-1
        frames = frames.astype(np.float32)
        # return tensors
        return torch.from_numpy(frames), torch.from_numpy(ecg)


def train_one_epoch(model, loader, optimizer, device, loss_fn=None):
    model.train()
    total_loss = 0.0
    if loss_fn is None:
        loss_fn = hybrid_loss
    for i, (frames, ecg) in enumerate(loader):
        frames = frames.to(device)
        ecg = ecg.to(device)
        with torch.set_grad_enabled(True):
            pred = model(frames)
            loss = loss_fn(pred, ecg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        if i >= 10:
            break
    return total_loss / (i+1)


def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }
    torch.save(state, path)


def apply_pruning(model, amount=0.2):
    """Placeholder: global unstructured pruning on conv and linear weights"""
    try:
        import torch.nn.utils.prune as prune
        modules = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                modules.append((m, 'weight'))
        # simple global unstructured pruning
        prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=amount)
        return True
    except Exception as e:
        print('Pruning failed:', e)
        return False


def apply_dynamic_quantization(model):
    """Apply dynamic quantization where supported (mainly for linear/embedding layers).
    Conv3d is not supported by dynamic quantization; this function is a placeholder.
    """
    try:
        qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        return qmodel
    except Exception as e:
        print('Dynamic quantization failed:', e)
        return model


def build_model(model_type='advanced', device=None):
    if device is None:
        device = torch.device('cpu')
    if model_type == 'mobile':
        return MobilePhysNet().to(device)
    elif model_type == 'efficient':
        return EfficientPhysNet().to(device)
    elif model_type == 'physformer':
        return PhysFormerNet().to(device)
    elif model_type == 'temporal_attention':
        return TemporalAttentionPhysNet().to(device)
    return AdvancedPhysNet().to(device)


def train(config=None):
    device = torch.device('cpu')
    model_type = 'advanced' if config is None else config.get('model_type', 'advanced')
    loss_fn = hybrid_loss if config is None else config.get('loss_fn', hybrid_loss)
    dataset = RPpgDataset(seq_len=config.get('seq_len', 32) if config else 32)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    model = build_model(model_type, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2 if config is None else config.get('epochs', 10)
    best_loss = 1e9
    out_path = CHECKPOINT_DIR / f'{model_type}_best.pth'
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, device, loss_fn=loss_fn)
        t1 = time.time()
        print(f'Epoch {epoch+1}/{epochs} loss={loss:.4f} time={t1-t0:.1f}s')
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, epoch+1, out_path)
    print('Training complete. Best loss:', best_loss)
    return model


def run_full_pipeline(train_epochs=2, prune_amount=0.2, finetune_epochs=1):
    """Train, prune, fine-tune, quantize pipeline (smoke-run friendly)."""
    device = torch.device('cpu')
    print('Starting full pipeline: train -> prune -> finetune -> quantize')
    # initial train
    model = AdvancedPhysNet().to(device)
    dataset = RPpgDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(train_epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f'[PIPELINE] Train epoch {epoch+1}/{train_epochs} loss={loss:.4f}')
        save_checkpoint(model, optimizer, epoch+1, CHECKPOINT_DIR / f'advanced_preprune_epoch{epoch+1}.pth')

    # apply pruning
    pruned_ok = apply_pruning(model, amount=prune_amount)
    if pruned_ok:
        print(f'[PIPELINE] Applied pruning amount={prune_amount}')
        # finetune
        opt2 = optim.Adam(model.parameters(), lr=5e-4)
        for e in range(finetune_epochs):
            loss = train_one_epoch(model, loader, opt2, device)
            print(f'[PIPELINE] Finetune epoch {e+1}/{finetune_epochs} loss={loss:.4f}')
        save_checkpoint(model, opt2, train_epochs + finetune_epochs, CHECKPOINT_DIR / 'advanced_pruned.pth')
    else:
        print('[PIPELINE] Pruning skipped or failed')

    # apply dynamic quantization (best-effort)
    try:
        qmodel = apply_dynamic_quantization(model)
        torch.save(qmodel.state_dict(), CHECKPOINT_DIR / 'advanced_quantized.pth')
        print('[PIPELINE] Dynamic quantization applied and saved')
    except Exception as e:
        print('[PIPELINE] Quantization failed:', e)

    return model


def smoke_test():
    print('Running smoke test...')
    model = AdvancedPhysNet()
    dataset = RPpgDataset()
    loader = DataLoader(dataset, batch_size=2)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    frames, ecg = next(iter(loader))
    out = model(frames)
    print('smoke forward shape:', out.shape)
    loss = negative_pearson_loss(out, ecg)
    print('smoke loss:', loss.item())


if __name__ == '__main__':
    smoke_test()
