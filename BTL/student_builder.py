"""Student builder for distillation dry-run."""
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
