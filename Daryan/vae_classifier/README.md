# VAE classifier: oil vs no oil (SAR chips)

A joint variational autoencoder + classifier in PyTorch. The encoder maps a SAR image chip to a
32-dimensional latent code z. The decoder reconstructs the chip from z, and a small MLP on z
predicts **Oil (1)** vs **No Oil (0)**. It uses the same train/val/test split as Josh's CNN
(`classifier_josh/split.json`), so the two models can be compared directly.

## Setup

The image folder `images/Oil Spills/` and the split file `classifier_josh/split.json` exist only
on the `josh-classifier` branch. This work lives on a branch made from it, `daryan-vae-classifier`.

```bash
git switch daryan-vae-classifier
pip install -r Daryan/vae_classifier/requirements.txt
```

All commands below are run from the **repo root**. Paths are relative and work on Windows and
macOS. The device is picked automatically (CUDA, then Apple MPS, then CPU). Use `--device cpu`
to force CPU.

## Commands

```bash
# 1. Quick end-to-end check (~200 train / 100 val images, 2 epochs, then saves and reloads the checkpoint)
python Daryan/vae_classifier/train.py --smoke-test --run-name smoke_test
python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/smoke_test --smoke-test --corruption all

# 2. Train (defaults: 128x128, latent 32, recon_weight 1, beta 1, lambda 1000, lr 3e-4, KL warm-up 3 epochs)
python Daryan/vae_classifier/train.py --epochs 20 --run-name vae_e20

# 3. Final evaluation on the held-out test split, plus simulated channel damage
python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/vae_e20 --corruption all

# 4. Evaluate on images that went through the compression + noisy-channel pipeline.
#    The folder needs "Oil/" and "No Oil/" subfolders. Test-split images are matched by file name
#    (any image extension, so .png output works).
python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/vae_e20 --data-dir path/to/decoded_images
#    ...or score every image in that folder (labelled by subfolder) instead of only the test split:
python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/vae_e20 --data-dir path/to/decoded_images --all-images

# 5. Variational information bottleneck (encoder + classifier + KL only, no decoder)
python Daryan/vae_classifier/train.py --recon-weight 0 --epochs 20 --run-name vib_e20

# 6. Data sanity checks (duplicates, padding, grayscale)
python Daryan/vae_classifier/audit_data.py
```

Run `python Daryan/vae_classifier/train.py -h` (or `evaluate.py -h`) for every option. The main ones:

| option | default | meaning |
|---|---|---|
| `--split-file` | `classifier_josh/split.json` | teammate's split; never created or modified here |
| `--img-size` | 128 | chips are resized from 400x400 |
| `--latent-dim` | 32 | size of z |
| `--recon-loss` | `bce` | `bce` or `mse` |
| `--recon-weight` | 1 (bce), 2 (mse) | 0 = VIB, decoder not run |
| `--beta` | 1 | KL weight after warm-up |
| `--lam` / `--lambda` | 1000 | classification weight |
| `--kl-warmup-epochs` | 3 | beta ramps linearly from 0 over this many epochs |
| `--lr`, `--batch-size`, `--epochs`, `--patience` | 3e-4, 64, 30, 8 | Adam; early stopping on val F1 |
| `--no-pool-features` | off | use the flatten-only encoder head (see "Changes from the original plan") |
| `--corruption` (evaluate) | `none` | `gaussian`, `jpeg`, `blocks` or `all` |

## Files

| file | contents |
|---|---|
| `model.py` | `VAEClassifier` (encoder, reparameterization, decoder, classifier head) and `load_checkpoint` |
| `losses.py` | reconstruction (BCE/MSE), KL, and the weighted joint loss |
| `data.py` | split loading and validation, in-memory dataset, augmentation, `--data-dir` matching, corruptions |
| `engine.py` | eval loop, active-unit count, per-term encoder gradient norms |
| `utils.py` | seeding, device choice, metrics |
| `train.py`, `evaluate.py` | the two entry points |
| `audit_data.py` | read-only data checks |

Each training run writes to `Daryan/vae_classifier/runs/<run-name>/`:
- `best.pt`: best checkpoint by validation F1
- `config.json`: all options, split file path and SHA-256, versions, git commit
- `metrics.csv`: one row per epoch, with each loss term raw and weighted, val metrics, active latent units, and encoder gradient norm per term
- `loss_curves.png`

`evaluate.py` writes to `runs/<run-name>/eval/<name>/`:
- `metrics.json`, `predictions.csv`
- `confusion_matrix.png`, `reconstructions.png`, `latent_projection.png` (PCA and t-SNE)
- with `--corruption`: `corruption_results.csv`, `corruption_curves.png`, `corruption_examples.png`

`runs/` is git-ignored.

## Model

```
x (1x128x128, in [0,1])
 -> 4 x [Conv 4x4 stride 2 -> BatchNorm -> LeakyReLU(0.2)]  channels 32, 64, 128, 256 -> 256x8x8
 -> [flatten (16384) ++ global avg pool (256) ++ global max pool (256)]
 -> Linear -> mu (32),  Linear -> log sigma^2 (32)
z = mu + sigma * eps  (training),   z = mu  (evaluation)
decoder:    Linear -> 256x8x8 -> 3 x [ConvTranspose -> BN -> LeakyReLU] -> ConvTranspose -> sigmoid
classifier: Linear(32, 64) -> LeakyReLU -> Linear(64, 1) logit
```

The model has 3.0 M parameters.

**Training details**
- Augmentation (train only): random horizontal and vertical flips and 90-degree rotations.
- Images are decoded once and kept in memory, so an epoch takes about 12 s on an Apple M2.
- The checkpoint is chosen on validation F1 (threshold 0.5), and only once KL warm-up has finished, so the saved model was trained with the full beta.
- Early stopping counts epochs without improvement after warm-up.
- `train.py` never loads the test images.

**Split handling**
- `data.load_split` checks that the file exists and is valid JSON with `train`/`val`/`test` lists of `[path, 0|1]`, that every image exists, and that no image appears twice. Otherwise it stops with an error.
- The split's SHA-256 is stored in `config.json` and in the checkpoint. `evaluate.py` prints a warning if you evaluate with a different split.

**Evaluation**
- `evaluate.py` reports metrics at threshold 0.5 and at the threshold that maximizes F1 on the validation split (never chosen on test).
- Cross-entropy is the plain unweighted `-log q(true label)`, in nats and bits.

## Loss and how the default weights were chosen

`L = recon_weight * Recon + beta * KL + lambda * BCE`. Every term is per image and averaged over the batch:
- **Recon:** pixel BCE summed over the 16,384 pixels, in nats.
- **KL:** summed over the 32 latent dimensions, in nats.
- **BCE:** classification cross-entropy with `pos_weight = n_neg/n_pos = 2586/1290 = 2.005` from the train split.

With `recon_weight = beta = 1` the first two terms are exactly the negative ELBO.

**Raw magnitudes are misleading.** Reconstruction is about 9,700 nats per image (0.59 nats per pixel), and most of that is radar speckle that no model can predict. KL is about 100 nats, and BCE is below 1 nat. The quantity that decides which term "dominates" is how strongly each term pushes the shared encoder. So `train.py` logs the encoder gradient norm of each weighted term every epoch (`gnorm_*` in `metrics.csv`).

- In a first calibration run with `lambda = 1`, the classification gradient was 300–1000x smaller than the reconstruction gradient. The classifier barely shaped the latent space.
- A sweep over lambda (12 epochs each, otherwise defaults; runs in `runs/sweep2_lam*`) at epoch 12:

| lambda | val F1 | val acc | val recon (nats/img) | KL | encoder grad recon / KL / BCE |
|---|---|---|---|---|---|
| 100 | 0.689 | 0.783 | 9635 | 52 | ~1e3 / 3e2 / 1e2–6e2 |
| 300 | 0.848 | 0.900 | 9620 | 67 | ~1e3 / 4e2 / 5e2–3e3 |
| **1000** | **0.880** | **0.924** | **9615** | 113 | **~1e3 / 5e2 / 2e3–9e3** |
| 3000 | 0.877 | 0.920 | 9680 | 156 | ~2e3 / 6e2 / 1e4–2.5e4 |

**lambda = 1000 is the default.**
- Reconstruction and classification gradients are of the same order.
- Reconstruction quality is not worse than at small lambda.
- Validation F1 is the best of the sweep.
- At 3000, classification gradients are about 10x reconstruction, reconstruction gets worse, and F1 does not improve.

**beta = 1** keeps the true ELBO, and all 32 latent units stay active, so there is no posterior collapse. The **3-epoch linear warm-up** avoids the large early KL spike seen in epoch 1.

**MSE default weight is 2.** The gradient of summed squared error with respect to the decoder's output logit is `2(s - x) s(1 - s)`, at most half of the BCE gradient `(s - x)`. Weight 2 matches BCE's pull at mid-gray. Measured encoder gradients with MSE were 1–3e3, the same range as BCE.

**VIB view.** With `--recon-weight 0` the objective is `1000·CE + 1·KL`, the same as `CE + 0.001·KL`. That is a typical variational information bottleneck beta.

## Results (run `vae_e20`, 20 epochs, best epoch 16)

The test split has 832 chips (277 Oil). Always predicting "No Oil" gives **0.667** accuracy.

| metric | t = 0.5 | t = 0.419 (best F1 on val) |
|---|---|---|
| accuracy | **0.933** | 0.930 |
| precision | 0.911 | 0.890 |
| recall | 0.884 | 0.903 |
| F1 | **0.897** | 0.896 |
| ROC-AUC | 0.981 | 0.981 |
| PR-AUC | 0.966 | 0.966 |
| mean cross-entropy | 0.169 nats = 0.243 bits | |
| confusion [[TN FP] [FN TP]] | [[531 24] [32 245]] | [[524 31] [27 250]] |

Loss terms at the best epoch (per image):

| | recon | KL | BCE | weighted: recon / β·KL / λ·BCE |
|---|---|---|---|---|
| train | 9773 | 90.5 | 0.252 | 9773 / 90.5 / 252 |
| val | 9623 | 90.4 | 0.277 | 9623 / 90.4 / 277 |

Encoder gradient norms at that epoch: recon 1.8e3, KL 4.6e2, BCE 3.4e3.

**Simulated channel damage (test split, threshold 0.5):**

| severity | Gaussian noise σ | acc / F1 | JPEG quality | acc / F1 | 8x8 blocks erased | acc / F1 |
|---|---|---|---|---|---|---|
| 0 | clean | 0.933 / 0.897 | clean | 0.933 / 0.897 | clean | 0.933 / 0.897 |
| 1 | 0.05 | 0.906 / 0.851 | 75 | 0.933 / 0.897 | 2% | 0.727 / 0.501 |
| 2 | 0.1 | 0.821 / 0.670 | 50 | 0.928 / 0.889 | 5% | 0.645 / 0.222 |
| 3 | 0.2 | 0.751 / 0.457 | 25 | 0.930 / 0.893 | 10% | 0.564 / 0.278 |
| 4 | 0.3 | 0.724 / 0.350 | 10 | 0.921 / 0.875 | 20% | 0.552 / 0.469 |
| 5 | 0.5 | 0.679 / 0.125 | 5 | 0.883 / 0.809 | 40% | 0.544 / 0.547 |

- **JPEG:** barely matters until quality 5.
- **Gaussian noise:** recall drops sharply. Noise hides the dark slicks, and the model answers "No Oil".
- **Erased blocks:** even 2% is very damaging. Black squares look like dark spots, which the model never saw in training. This is the corruption closest to lost packets, so the classifier probably needs erasure-style augmentation (or the channel needs protection) before compressed images will classify well.

Corruptions are applied to the full 400x400 image before resizing, so 8x8 blocks line up with the JPEG/DCT block grid. Each image's corruption is seeded, so runs are reproducible.

**Reconstructions are blurry.** The decoder recovers overall brightness and large structures but not thin slicks or speckle (see `reconstructions.png`). The classification information in z therefore comes mostly from the lambda term. The PCA and t-SNE plots of the test mu vectors separate the classes well.

## Data issues found

From `audit_data.py` and from inspecting the confident test errors:

1. **Swath-edge padding correlates with the label.**
   - 38% of No Oil chips contain white no-data padding, versus 11% of Oil chips.
   - 285 of the 289 chips that are more than half padding are No Oil.
   - The model uses this shortcut. On test, Oil recall is **0.74 on padded chips vs 0.90 on unpadded ones**, and No Oil specificity is 0.99 vs 0.94.
   - Possible fixes: mask or fill padding before training, or report metrics on unpadded chips separately.
2. **Near-duplicates across splits.**
   - There are 9 near-identical pairs: the same scene with different brightness or contrast, usually with consecutive file numbers. There are no exact duplicates.
   - 6 pairs cross splits: 5 train↔val and 1 train↔test (`class_1_01243` train vs `class_1_01242` test).
   - This leakage is small, but the split owner may want to know. The list is in `runs/data_audit/near_duplicates.csv`.
3. **Chips that may be mislabelled** (confident model errors, worth a human look):
   - Labelled Oil but nearly blank at this scale: `class_1_01113` (p(oil) = 0.015), `class_1_00119`, `class_1_00982`.
   - Labelled No Oil but showing slick-like dark streaks or patches: `class_0_02727` (p = 0.95), `class_0_01666` (0.94), `class_0_01602`, `class_0_01193`. These may also be look-alikes (low wind, ship wakes), which are exactly what makes this task hard.
4. **Other observations.**
   - All 5,538 chips are 400x400 RGB with three identical channels (true grayscale).
   - Some No Oil chips are almost entirely padding (17 are more than 90% padding), and a few are saturated or contain land-mask cut-outs.
   - Many Oil chips are low-contrast with thin, faint slicks, which is one reason to try `--img-size 256`.

## Comparing with Josh's CNN

- **Same split:** check the `split_sha256` in `config.json`.
- **Same metrics:** use the threshold-0.5 numbers. `predictions.csv` has per-image probabilities for paired comparisons (e.g. McNemar's test).
- **Different input pipelines:**
  - Josh: pretrained ResNet-18, 224x224, 3-channel, ImageNet normalization.
  - This model: from scratch, 128x128, 1-channel, [0, 1].
- **Class weighting:** both use pos_weight of about 2.0.

## Changes from the original plan, and why

- **Pooled features in the encoder head (default on).** With a flatten-only head, even a plain classifier on this encoder reached only ~0.72 train accuracy after 12 epochs. A logistic regression on 7 simple image statistics did better (val ROC-AUC 0.886). Adding global average and max pooling of the last feature map to the inputs of the mu/log σ² heads raised that to 0.91 train / 0.905 val accuracy. The flattened map is kept, so the decoder still gets spatial information. `--no-pool-features` restores the original head.
- **Learning rate 3e-4** instead of 1e-3. At 1e-3 validation metrics swung wildly between epochs, and training stalled.
- **Split file name.** The file is `classifier_josh/split.json` (the original brief said `spit.json`).

## Notes

- **Reproducibility.** Runs are seeded. Two runs with the same options gave identical per-epoch numbers on MPS. Bit-exact results across different hardware are not guaranteed.
- **t-SNE in a separate process.** On some macOS/conda setups, PyTorch's and scikit-learn's OpenMP libraries crash when loaded in the same process. So t-SNE runs in its own process; if it fails, it is skipped with a message.
