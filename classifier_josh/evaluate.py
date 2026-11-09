"""
evaluate.py - run the best saved model once on the test images.
Reports the same metrics as Daryan's classifier (threshold 0.5)
"""
 
import torch
 
from data import make_loaders
from train import make_model, CHECKPOINT, DEVICE
 
# load best model
model = make_model()
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()   # checking mode, no learning
 
# Run every test image through it once
_, _, test_loader = make_loaders()
probs, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        probs.append(torch.sigmoid(model(x.to(DEVICE)).squeeze(1)).cpu()) # P(oil)
        labels.append(y)
p, y = torch.cat(probs), torch.cat(labels)
pred = (p >= 0.5).float()   # oil when P(oil) >= 0.5
 
# Count the four kinds of outcome
tp = ((pred == 1) & (y == 1)).sum().item()       # said oil, was oil
fp = ((pred == 1) & (y == 0)).sum().item()       # said oil, wasn't
fn = ((pred == 0) & (y == 1)).sum().item()       # said no oil, was oil
tn = ((pred == 0) & (y == 0)).sum().item()       # said no oil, wasn't
 
accuracy = (tp + tn) / len(y)
precision = tp / max(tp + fp, 1)        # when it says oil, how often right?
recall = tp / max(tp + fn, 1)               # of the real spills, how many were caught
f1 = 2 * tp / max(2 * tp + fp + fn, 1)
 
# ROC-AUC: chance random oil image gets a higher P(oil) than random no-oil image
pos, neg = p[y == 1], p[y == 0]
auc = ((pos[:, None] > neg[None, :]).float().mean()
       + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item()
 
print(f"test images: {len(y)} ({int(y.sum())} oil)")
print(f"accuracy  {accuracy:.3f}")
print(f"precision {precision:.3f}")
print(f"recall    {recall:.3f}")
print(f"F1        {f1:.3f}")
print(f"ROC-AUC   {auc:.3f}")
print(f"always 'no oil' accuracy: {(y == 0).float().mean().item():.3f}")
print()
print("confusion matrix      predicted no oil   predicted oil")
print(f"  actually no oil     {tn:16d} {fp:15d}")
print(f"  actually oil        {fn:16d} {tp:15d}")