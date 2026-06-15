"""Run a minimal distillation dry-run without torchvision by synthesizing data."""
import torch
from torch.utils.data import DataLoader, TensorDataset
import importlib.util
from pathlib import Path

# import distill module by file path to avoid module resolution issues
distill_path = Path(__file__).resolve().parent.parent / 'distill.py'
spec_d = importlib.util.spec_from_file_location('distill_mod', str(distill_path))
distill = importlib.util.module_from_spec(spec_d)
spec_d.loader.exec_module(distill)

teacher_path = Path('ket_qua') / 'fake_teacher.pt'
student_module_path = Path('student_builder.py')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load teacher
teacher = torch.jit.load(str(teacher_path)).to(device).eval()
# import student builder
spec = importlib.util.spec_from_file_location('student_mod', str(student_module_path))
student_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_mod)
student = student_mod.build_student().to(device)

# synthetic dataset: images 224x224, num_classes=10
N = 64
x = torch.randn(N, 3, 224, 224)
y = torch.randint(0, 10, (N,))
loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, num_workers=0, pin_memory=False)

optim = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=0.0)

loss = distill.train_one_epoch(student, teacher, loader, optim, device, temperature=4.0, alpha=0.5)
print('Dry-run train loss:', loss)
torch.save(student.state_dict(), 'ket_qua/student_dryrun.pt')
print('Saved student state to ket_qua/student_dryrun.pt')
