"""
train.py - the classifier and how it learns.
  make_model() creates f_theta, which turns an image into one "oil" score (logit z)
  loss_fn() is the governing equation: how wrong the scores are (cross-entropy, with optional Renyi alpha)
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
 
 
# ---------------- Model ----------------
 
def make_model():
    """Pretrained ResNet-18 with its last layer swapped for a single 'oil' score."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # already learned to "see" on ImageNet
    model.fc = nn.Linear(model.fc.in_features, 1)                     # 512 features -> 1 score z
    return model
 
 
# ---------------- Loss ----------------
 
POS_WEIGHT = 3695 / 1843   # about 2: each oil image counts double as there are half as many
 
 
def loss_fn(z, y, alpha=1.0):
    """
    z: scores from the model, shape [batch, 1].   y: true labels, 1 = oil, 0 = no oil.
    alpha = 1   -> Shannon cross-entropy, the standard loss (average of -log q over the batch)
    alpha != 1  -> Renyi version over the batch; it turns back into the Shannon one as alpha goes to 1
    """
    # Sigmoid (2-class softmax) + log, in one step:
    #   oil image    (y=1): log P(oil)    = logsigmoid(z)
    #   no-oil image (y=0): log P(no oil) = logsigmoid(-z)
    log_q = F.logsigmoid((2 * y - 1) * z.squeeze(1))   # log of the probability given to the TRUE answer
    w = 1 + (POS_WEIGHT - 1) * y                         # weight: oil images POS_WEIGHT, no-oil images 1
 
    if alpha == 1:
        return -(w * log_q).sum() / w.sum()             # L = weighted average of -log q, instead of dividing by N
 
    q = torch.exp(log_q)
    return torch.log((w * q ** (alpha - 1)).sum() / w.sum()) / (1 - alpha)