Quickstart - BTL repository (web_cam entry)

Tổng quan:
- `web_cam.py` là entry chính với nhiều wrapper và subcommand:
  - `--show-cpu` : in thông tin CPU và profile (adaptive_runtime)
  - `--assess-image <path>` : đánh giá chất lượng ảnh (signal_quality)
  - `--assess-signal <npy|csv>` : đánh giá chất lượng tín hiệu 1D
  - `--adaptive-demo` : chạy demo `AdaptiveInferenceManager`
  - Subcommands: `export` và `distill` (chuyển tiếp tới `export_models.py` và `distill.py`)

Ví dụ nhanh (fake models được tạo sẵn bằng script trong `tools/`):

1) Tạo model fake (đã tạo sẵn bởi tôi):

```bash
python tools/create_fake_models.py
```

2) Export TorchScript (ví dụ):

```bash
python export_models.py --model ket_qua/fake_teacher.pt --format torchscript --out ket_qua/fake_exported.pt --example-size 1,3,224,224
# hoặc dùng wrapper
python -m web_cam export -- --model ket_qua/fake_teacher.pt --format torchscript --out ket_qua/fake_exported.pt --example-size 1,3,224,224
```

3) Distill dry-run (sử dụng student_builder.py và dữ liệu tổng hợp):

```bash
python tools/distill_dryrun.py
# hoặc dùng wrapper (yêu cầu torchvision) - nếu thiếu, dùng tools/distill_dryrun.py
```

4) Kiểm tra nhanh và chạy webcam (màn hình help):

```bash
python -m web_cam --help
python -m web_cam --adaptive-demo
```

Ghi chú:
- Tôi đã tách logic `main()` trong `web_cam.py` để giảm độ phức tạp.
- File `scripts/compile_exclude.py` dùng để kiểm tra biên dịch loại trừ `.venv` và `gpt`.

Tiếp theo tôi có thể:
- Sửa các Sonar warnings còn lại (giảm complexity các hàm lớn khác), và/hoặc
- Sửa các file trong `gpt/` để loại bỏ lỗi syntax/indentation, hoặc bỏ chúng khỏi kiểm tra.

Nếu bạn có model thực muốn tôi dùng để export/distill, gửi đường dẫn file model (.pt hoặc tương tự).