# Knowledge Distillation for Lightweight Models (Fashion-MNIST)

UML501 Machine Learning Lab project 
Authors: Armandeep Singh Taggar (1024030219), Shrasthi Singh Rathore(1024031049)

## Problem
Can a compact student network learn from a larger teacher and outperform the same student trained only on ground-truth labels?

## Method
1. Train a larger **teacher** CNN (3,584,650 parameters) on Fashion-MNIST.
2. Train small **student** CNNs (three sizes) using labels only (baseline).
3. Train the same students with distillation:
   `L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1 - alpha) * CE(z_s, y)`
4. Compare accuracy, macro-F1, parameter count, file size and CPU latency; sweep temperature (T) and loss weight (alpha).

## Dataset
Fashion-MNIST (Xiao et al., 2017): 70,000 grayscale 28x28 images, 10 balanced classes. Source: https://github.com/zalandoresearch/fashion-mnist
Split: 55,000 train / 5,000 validation / 10,000 test (seed 42). Normalisation statistics are computed on the training split only. The dataset downloads automatically on first run.

## Repository contents
| File | Description |
|---|---|
| `kd_fashion_mnist.py` | Complete pipeline: EDA, models, training, distillation, experiments, plots |
| `figures/` | EDA plots and result figures (`viz1` to `viz4`) |
| `results_table.csv` | Summary table (mean over 3 seeds) |
| `results_all_runs.csv` | Every individual run |
| `results_temperature.csv`, `results_alpha.csv` | Hyperparameter sweeps |
| `report` | Final project report (Word file, if included in the repository) |

## How to run

**Google Colab:** upload `kd_fashion_mnist.py`, then run `%run kd_fashion_mnist.py` in a cell. Download `figures/` and `results_*.csv` before the session ends.

**Local:**
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
python kd_fashion_mnist.py
```

## Experimental setup
- Epochs: 15 (teacher and students). Seeds: 0, 1, 2 for students; seed 42 for the data split and teacher.
- Optimizer: Adam, learning rate 2e-3 with cosine schedule, batch size 128, no data augmentation.
- Main distillation setting: T = 4, alpha = 0.7. Best epoch chosen on the validation set; test set used only for final reporting.
- Hardware and software: Google Colab CPU runtime (1 thread), torch 2.11.0+cpu, Python *(add version)*. Total run time about 3 hours.

## Results (test set, mean over 3 seeds)

| Model | Mode | Accuracy | Std | Macro-F1 | Params | Size (KB) | Latency (ms/img) |
|---|---|---|---|---|---|---|---|
| Teacher | labels | 0.9346 | n/a (1 run) | 0.9345 | 3,584,650 | 14,013.8 | 3.81 |
| Student-tiny | labels | 0.8895 | 0.0012 | 0.8888 | 6,818 | 32.7 | 0.28 |
| Student-tiny | distill | 0.8890 | 0.0050 | 0.8887 | 6,818 | 32.7 | 0.29 |
| Student-small | labels | 0.9073 | 0.0036 | 0.9068 | 26,746 | 110.7 | 0.36 |
| Student-small | distill | 0.9094 | 0.0039 | 0.9088 | 26,746 | 110.7 | 0.40 |
| Student-medium | labels | 0.9170 | 0.0001 | 0.9167 | 105,962 | 420.1 | 0.54 |
| Student-medium | distill | 0.9225 | 0.0018 | 0.9225 | 105,962 | 420.1 | 0.55 |

Latency is single-image CPU inference on Colab (1 thread) and is noisy; small differences between students of the same size are measurement noise.

### Gain from distillation per seed (accuracy points, distill minus labels)

| Student | Seed 0 | Seed 1 | Seed 2 | Mean +/- std | Paired t-test p |
|---|---|---|---|---|---|
| Medium | +0.71 | +0.34 | +0.61 | +0.55 +/- 0.19 | 0.038 |
| Small | +0.36 | +0.06 | +0.22 | +0.21 +/- 0.15 | 0.133 |
| Tiny | +0.36 | +0.24 | -0.77 | -0.06 +/- 0.62 | 0.889 |

With only 3 seeds (df = 2) the p-values are indicative, not conclusive.

### Hyperparameter sweeps (small student, mean over 3 seeds)

| Temperature T (alpha = 0.7) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Accuracy | 0.9065 | 0.9091 | 0.9094 | 0.9092 | 0.9026 |

| Alpha (T = 4) | 0.00 | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|---|
| Accuracy | 0.9073 | 0.9085 | 0.9071 | 0.9074 | 0.9080 |

The labels-only small student scores 0.9073. Alpha = 0 reduces the loss to plain cross-entropy, so it reproduces that baseline exactly (a sanity check).

## Key findings
- **Efficiency:** compared with the teacher, the students have 34x (medium), 134x (small) and 526x (tiny) fewer parameters, files that are 33x to 428x smaller, and are roughly 7x to 14x faster on CPU. The accuracy cost of the distilled students is 1.2 points (medium), 2.5 points (small) and 4.6 points (tiny).
- **Distillation helps modestly and only for larger students:** consistent gain for the medium student (+0.55 points, positive in all 3 seeds), a small and weaker gain for the small student (+0.21), and no reliable gain for the tiny student (mean -0.06, one seed worse).
- **Temperature:** T between 2 and 8 gave similar results, slightly above the baseline; T = 16 was worse than the baseline.
- **Alpha:** no clear effect; all differences are within seed variation.
- **Failure cases:** Shirt is the hardest class (confused with T-shirt, Pullover and Coat), consistent with the EDA similarity analysis. In the seed-0 confusion matrices, Pullover recall rose from 834 to 882 out of 1,000 with distillation.

![Accuracy vs model size](figures/viz2_acc_vs_size.png)
![Temperature and alpha](figures/viz3_temperature_alpha.png)
![Confusion matrices](figures/viz4_confusion.png)

## Limitations
- Only 3 seeds and one dataset; Fashion-MNIST is an easy benchmark, so differences between models are small.
- Pairing across modes is approximate: student weights are initialised before the seed is set inside the training function, so labels-only and distilled runs of the same seed share batch order but not exactly the same initial weights.
- The teacher reached a training loss near 0.005, so its soft targets on training images may be close to one-hot, which could limit what the student gains (a hypothesis, not tested).
- Latency was measured on a shared single-thread Colab CPU and varies between runs.
- Confusion matrices come from a single seed and should be read as illustrative.

## References
- Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network.
- Xiao, Rasul, Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
