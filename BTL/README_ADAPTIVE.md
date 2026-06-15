Adaptive runtime and inference manager

Quick start

1. Install additional optional dependencies:

pip install -r requirements_adaptive.txt

2. Run the lightweight tests:

python test_adaptive.py

Notes
- The code is dependency-light: it falls back to returning model paths if PyTorch/ONNX runtime are not available.
- For production, prepare TorchScript or ONNX exports of your models and pass their paths to the manager.

Exporting models

 - TorchScript:

	 ```bash
	 python export_models.py --model path/to/train_model.pt --format torchscript --out model_ts.pt --example-size 1,3,224,224
	 ```

 - ONNX:

	 ```bash
	 python export_models.py --model path/to/train_model.pt --format onnx --out model.onnx --example-size 1,3,224,224
	 ```

 - Dynamic quantization (CPU-friendly):

	 ```bash
	 python export_models.py --model path/to/train_model.pt --format quantize --out model_quant.pt
	 ```

Notes: installs for ONNX export and simplification are optional; the scripts fall back gracefully if dependencies are missing.
