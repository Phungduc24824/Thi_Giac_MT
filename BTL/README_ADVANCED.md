Advanced export & distillation workflows

Overview

This document contains quick recipes for exporting models to TorchScript/ONNX and for performing knowledge distillation and edge deployment (OpenVINO/TensorRT/TFLite).

Export to TorchScript

- Ensure model is in eval mode and example input matches expected shape/dtype.
- Use `torch.jit.trace(model, example)` or `torch.jit.script(model)` and save with `.save()`.

Export to ONNX

- Use `torch.onnx.export(model, example, out_path, opset_version=11, input_names=['input'], output_names=['output'])`.
- Validate the ONNX model with `onnx.checker` and `onnxruntime.InferenceSession`.

OpenVINO / TensorRT

- OpenVINO: run Model Optimizer `mo` to convert ONNX -> IR (XML + BIN). Then use the OpenVINO runtime to load and run.
- TensorRT: either use NVIDIA's `trtexec` CLI to build an engine from ONNX or integrate TensorRT builder in Python.

Knowledge Distillation (PyTorch template)

See `distill.py` for a minimal teacher-student training loop using soft targets (temperature) and a combined loss.

Edge models and optimizations

- Consider EfficientNet-lite, MobileNetV3, MnasNet, GhostNet for mobile/edge.
- Quantization: dynamic (post-training) or static (calibration) quantization reduces model size and improves CPU inference.
- Pruning and structured sparsity can help reduce FLOPs; re-evaluate accuracy.

Benchmarking

- Measure latency and throughput on target device.
- Save benchmark results in `ket_qua/` with clear metadata (model, device, config).

References

- ONNX: https://onnx.ai
- OpenVINO: https://docs.openvino.ai
- TensorRT: https://developer.nvidia.com/tensorrt
- Distillation paper: Hinton et al., 2015
