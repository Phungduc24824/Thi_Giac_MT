"""Model utilities: loading and optional export helpers for TorchScript/ONNX.
Lightweight wrappers that don't hard-require heavy deps at import time.
"""
from typing import Any, Optional
import logging

logging.basicConfig(level=logging.INFO)


def load_torch_model(path: str, map_location: str = 'cpu') -> Any:
    try:
        import torch
    except Exception:
        logging.warning('PyTorch not installed. Returning path string.')
        return path

    try:
        # prefer JIT if possible; avoid calling torch.load on untrusted pickles
        try:
            m = torch.jit.load(path, map_location=map_location)
            logging.info('Loaded TorchScript model from %s', path)
            return m
        except Exception:
            logging.warning('torch.jit.load failed; not attempting torch.load for security. Returning path string.')
            return path
    except Exception as e:
        logging.exception('Failed to load torch model: %s', e)
        return path


def load_onnx_model(path: str) -> Optional[Any]:
    try:
        import onnxruntime as ort
    except Exception:
        logging.warning('onnxruntime not installed. Returning None.')
        return None
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        logging.info('Loaded ONNX model %s', path)
        return sess
    except Exception:
        logging.exception('Failed to load ONNX model %s', path)
        return None


def export_torchscript(model: Any, example_input: Any, out_path: str) -> bool:
    try:
        import torch
    except Exception:
        logging.warning('PyTorch not installed; cannot export.')
        return False
    try:
        scripted = torch.jit.trace(model, example_input)
        scripted.save(out_path)
        logging.info('Exported TorchScript to %s', out_path)
        return True
    except Exception:
        logging.exception('TorchScript export failed')
        return False


def export_onnx(model: Any, example_input: Any, out_path: str, input_names=None, output_names=None) -> bool:
    try:
        import torch
    except Exception:
        logging.warning('PyTorch not installed; ONNX export may still work if model provides export API.')
    try:
        import torch
        torch.onnx.export(model, example_input, out_path, input_names=input_names, output_names=output_names)
        logging.info('Exported ONNX to %s', out_path)
        return True
    except Exception:
        logging.exception('ONNX export failed')
        return False
