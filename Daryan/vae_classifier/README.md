# VAE classifier: oil vs no oil

A joint variational autoencoder and classifier in PyTorch for SAR chips (Oil = 1, No Oil = 0). A conv encoder maps each 128×128 grayscale chip to a 32-dimensional latent code. A decoder reconstructs the chip from that code, and a small MLP predicts the label. The loss is `recon + β·KL + λ·BCE`, with defaults 1, 1 and 1000. It uses Josh's split (`classifier_josh/split.json`), so its results compare directly with his CNN.

```bash
pip install -r Daryan/vae_classifier/requirements.txt
python Daryan/vae_classifier/train.py --epochs 20 --run-name vae_e20
python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/vae_e20 --corruption all
```

Run the commands from the repo root. They need `images/Oil Spills/` and `classifier_josh/split.json`, which are currently on the `josh-classifier` branch. Add `--smoke-test` for a quick check, or `-h` to see all options. Outputs go to `Daryan/vae_classifier/runs/`, which git ignores.

**Results** (test split, 832 chips): 93.3% accuracy, F1 0.897 and ROC-AUC 0.981, versus 66.7% for always guessing No Oil. JPEG compression barely affects it. Gaussian noise and erased 8×8 blocks hurt it a lot.

**Data caveats:** No Oil chips are much more likely than Oil chips to contain white padding at the swath edge, and the model partly uses this as a shortcut. There are also a few near-duplicate images across splits and some chips that may be mislabelled. `audit_data.py` lists them.
