"""Create a small TorchScript teacher model and a student builder module for dry-runs."""
from pathlib import Path
import torch
import torch.nn as nn

out_dir = Path(__file__).resolve().parent.parent / 'ket_qua'
out_dir.mkdir(exist_ok=True)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(8, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = SmallCNN()
# trace with example input (1,3,224,224)
example = torch.randn(1,3,224,224)
traced = torch.jit.trace(model, example)
teacher_path = out_dir / 'fake_teacher.pt'
traced.save(str(teacher_path))
print('Saved teacher TorchScript to', teacher_path)

# create a minimal student_builder.py at repo root
student_code = '''"""Student builder for distillation dry-run."""
import torch
import torch.nn as nn

def build_student():
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(4, 10)
    )
'''
student_path = Path(__file__).resolve().parent.parent / 'student_builder.py'
student_path.write_text(student_code)
print('Wrote student builder to', student_path)
