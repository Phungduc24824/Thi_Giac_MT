"""Export and quantize utilities for models: TorchScript, ONNX, dynamic quantization.

Usage examples:
  python export_models.py --model model.pt --format torchscript --out model_ts.pt --example-size 1,3,224,224
  python export_models.py --model model.pt --format onnx --out model.onnx --example-size 1,3,224,224
  python export_models.py --model model.pt --format quantize --out model_quant.pt
"""
import argparse
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)


def parse_size(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(','))


def export_torchscript(model_path: str, out_path: str, example_size=(1, 3, 224, 224)) -> bool:
    try:
        import torch
        from model_utils import load_torch_model
    except Exception:
        logging.exception('PyTorch or model_utils unavailable')
        return False
    model = load_torch_model(model_path)
    if isinstance(model, str):
        logging.error('Model loader returned path placeholder; cannot export')
        return False
    model.eval()
    example = torch.randn(example_size)
    try:
        try:
            traced = torch.jit.trace(model, example)
        except Exception:
            traced = torch.jit.script(model)
        traced.save(out_path)
        logging.info('Saved TorchScript to %s', out_path)
        return True
    except Exception:
        logging.exception('TorchScript export failed')
        return False


def export_onnx(model_path: str, out_path: str, example_size=(1, 3, 224, 224), opset=11) -> bool:
    try:
        import torch
        from model_utils import load_torch_model
    except Exception:
        logging.exception('PyTorch or model_utils unavailable')
        return False
    model = load_torch_model(model_path)
    if isinstance(model, str):
        logging.error('Model loader returned path placeholder; cannot export')
        return False
    model.eval()
    example = torch.randn(example_size)
    try:
        torch.onnx.export(model, example, out_path, opset_version=opset, input_names=['input'], output_names=['output'])
        logging.info('Saved ONNX to %s', out_path)
        return True
    except Exception:
        logging.exception('ONNX export failed')
        return False


def quantize_dynamic(model_path: str, out_path: str) -> bool:
    try:
        import torch
        from model_utils import load_torch_model
        from torch.quantization import quantize_dynamic
    except Exception:
        logging.exception('PyTorch or quantization APIs unavailable')
        return False
    model = load_torch_model(model_path)
    if isinstance(model, str):
        logging.error('Model loader returned path placeholder; cannot quantize')
        return False
    try:
        q_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        try:
            import io
            torch.jit.save(torch.jit.script(q_model), out_path)
        except Exception:
            torch.save(q_model.state_dict(), out_path)
        logging.info('Saved quantized model to %s', out_path)
        return True
    except Exception:
        logging.exception('Quantization failed')
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--format', choices=['torchscript', 'onnx', 'quantize'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--example-size', default='1,3,224,224')
    p.add_argument('--opset', type=int, default=11)
    args = p.parse_args()

    example_size = parse_size(args.example_size)
    if args.format == 'torchscript':
        ok = export_torchscript(args.model, args.out, example_size)
    elif args.format == 'onnx':
        ok = export_onnx(args.model, args.out, example_size, opset=args.opset)
    else:
        ok = quantize_dynamic(args.model, args.out)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
