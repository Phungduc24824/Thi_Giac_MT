"""Minimal knowledge distillation training loop (PyTorch).

Usage (example):
    python distill.py --teacher teacher.pt --student student.py --data dataset_dir --out student_distilled.pt

This is a template — adapt model imports and dataset to your project.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def distillation_loss(student_logits, teacher_logits, targets, temperature=4.0, alpha=0.5):
    # soft targets loss (KL-divergence) + hard label loss
    t = float(temperature)
    p_student = F.log_softmax(student_logits / t, dim=1)
    p_teacher = F.softmax(teacher_logits / t, dim=1)
    loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (t * t)
    loss_ce = F.cross_entropy(student_logits, targets)
    return alpha * loss_ce + (1.0 - alpha) * loss_kd


def train_one_epoch(student, teacher, loader, optim, device, temperature, alpha):
    student.train()
    teacher.eval()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student(x)
        loss = distillation_loss(s_logits, t_logits, y, temperature=temperature, alpha=alpha)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--student-module', required=True, help='Python module path providing build_student()')
    p.add_argument('--data', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', '--temp', dest='temperature', type=float, default=4.0,
                   help='Distillation temperature for soft targets')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='Weight decay for optimizer (L2 regularization)')
    p.add_argument('--out', default='student_distilled.pt')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load teacher: require TorchScript (.pt) to avoid untrusted pickle loading
    if not args.teacher.endswith('.pt'):
        raise SystemExit('Please provide a TorchScript teacher model (.pt) created by torch.jit.trace/script')
    teacher = torch.jit.load(args.teacher)
    teacher.to(device)
    teacher.eval()

    # dynamic import student builder
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location('student_mod', args.student_module)
    student_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_mod)
    student = student_mod.build_student()
    student.to(device)

    # Dataset placeholder — user should replace with real dataset
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = datasets.FakeData(transform=transform, size=1024, image_size=(3,224,224), num_classes=10)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)

    optim = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train_one_epoch(student, teacher, loader, optim, device, args.temperature, args.alpha)
        print(f'Epoch {epoch+1}/{args.epochs} loss={loss:.4f}')

    torch.save(student.state_dict(), args.out)
    print('Saved distilled student to', args.out)


if __name__ == '__main__':
    main()
