"""
train.py - the classifier and how it learns.
  make_model() creates f_theta, which turns an image into one "oil" score (logit z)
  loss_fn() is the governing equation: how wrong the scores are (cross-entropy, with optional Renyi alpha)
"""
from pathlib import Path
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

# ---------------- Training ----------------
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # GPU or CPU if there is none
CHECKPOINT = Path(__file__).parent / "best.pt"      # save best model
 
 
def run_epoch(model, loader, alpha=1.0, opt=None):
    """One pass over a loader. With an optimizer it trains, without one it only measures.
    Returns (average loss, accuracy, F1) using threshold P(oil) >= 0.5."""
    training = opt is not None
    model.train(training)
    losses, probs, labels = [], [], []
 
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = model(x)                       # scores
            loss = loss_fn(z, y, alpha)        # governing equation
            if training:
                opt.zero_grad()         # clear old gradients
                loss.backward()                # gradient of the loss for every weight
                opt.step()       #  nudge every weight down
            losses.append(loss.item() * len(y))
            probs.append(torch.sigmoid(z.squeeze(1)).detach().cpu())
            labels.append(y.cpu())
 
    p, y = torch.cat(probs), torch.cat(labels)
    pred = (p >= 0.5).float()      # oil when P(oil) >= 0.5
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    acc = (pred == y).float().mean().item()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return sum(losses) / len(y), acc, f1
 
 
if __name__ == "__main__":
    import argparse
    from data import make_loaders
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)        # the network is already pretrained
    parser.add_argument("--alpha", type=float, default=1.0)      # 1 = Shannon cross-entropy
    parser.add_argument("--overfit", action="store_true",        # can it memorize one batch
                        help="train on a single batch only, to check everything is wired correctly")
    args = parser.parse_args()
 
    torch.manual_seed(0)
    train_loader, val_loader, _ = make_loaders()      
    model = make_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device: {DEVICE}, alpha: {args.alpha}")
 
    if args.overfit:
        x, y = next(iter(train_loader))
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.train()
        for step in range(30):
            loss = loss_fn(model(x), y, args.alpha)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 5 == 0 or step == 29:
                print(f"step {step:2d}: loss {loss.item():.4f}")
        raise SystemExit
 
    best_f1 = -1
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, args.alpha, opt)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, args.alpha)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f}")
        if va_f1 > best_f1:               # keep the epoch with the best validation F1
            best_f1 = va_f1
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"           saved new best model (val F1 {va_f1:.3f})")
 
    print(f"done. best val F1 {best_f1:.3f}, model saved to {CHECKPOINT.name}")