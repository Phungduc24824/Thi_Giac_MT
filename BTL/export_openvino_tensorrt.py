"""Helpers to export ONNX and provide hooks for OpenVINO/TensorRT conversion.

These functions are convenience wrappers — actual conversion tools (mo, trtexec) are external CLIs.
"""
import subprocess
import logging
import os

logging.basicConfig(level=logging.INFO)


def export_to_onnx(model, example_input, out_path, opset=11, input_names=None, output_names=None):
    try:
        import torch
        model.eval()
        torch.onnx.export(model, example_input, out_path, opset_version=opset, input_names=input_names or ['input'], output_names=output_names or ['output'])
        logging.info('Exported ONNX to %s', out_path)
        return True
    except Exception:
        logging.exception('ONNX export failed')
        return False


def onnx_to_openvino(onnx_path, output_dir):
    """Call OpenVINO Model Optimizer if available (`mo` command)."""
    if not os.path.exists(onnx_path):
        logging.error('ONNX path not found: %s', onnx_path)
        return False
    try:
        cmd = ['mo', '--input_model', onnx_path, '--output_dir', output_dir]
        logging.info('Running: %s', ' '.join(cmd))
        subprocess.check_call(cmd)
        logging.info('OpenVINO IR generated in %s', output_dir)
        return True
    except Exception:
        logging.exception('OpenVINO conversion failed (ensure OpenVINO is installed and `mo` is on PATH)')
        return False


def onnx_to_tensorrt(onnx_path, out_engine_path, max_batch=1):
    """Try to build TensorRT engine via `trtexec` CLI if available."""
    if not os.path.exists(onnx_path):
        logging.error('ONNX path not found: %s', onnx_path)
        return False
    try:
        cmd = ['trtexec', '--onnx=' + onnx_path, '--saveEngine=' + out_engine_path, f'--batch={max_batch}']
        logging.info('Running: %s', ' '.join(cmd))
        subprocess.check_call(cmd)
        logging.info('TensorRT engine saved to %s', out_engine_path)
        return True
    except Exception:
        logging.exception('TensorRT conversion failed (ensure TensorRT trtexec is installed and on PATH)')
        return False
