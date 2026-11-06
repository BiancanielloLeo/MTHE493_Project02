"""
train.py - the classifier and how it learns.
   make_model() builds f_theta, which turns an image into one "oil" score (a logit z)
"""

import torch.nn as nn
from torchvision import models


# ---------------- Part 3: model ----------------

def make_model():
    """Pretrained ResNet-18 with its last layer swapped for a single 'oil' score."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # already learned to "see" on ImageNet
    model.fc = nn.Linear(model.fc.in_features, 1)                     # 512 features -> 1 score z
    return model