# Knowledge Distillation for Lightweight Models (Fashion-MNIST)

UML501 Machine Learning Lab project (Project 4)

## Problem
Can a compact student network learn from a larger teacher and outperform the same student trained only on ground-truth labels?

## What this project does
1. Trains a larger **teacher** CNN on Fashion-MNIST.
2. Trains a small **student** CNN using labels only (baseline).
3. Trains the same student using the teacher's soft targets with a distillation loss:
   `L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1 - alpha) * CE(z_s, y)`
4. Compares the models on accuracy, macro-F1, parameter count, file size and CPU inference latency.
5. Runs experiments varying student size, temperature (T) and loss weight (alpha).

## Dataset
Fashion-MNIST (Xiao et al., 2017): 70,000 grayscale 28x28 images, 10 balanced classes.
Source: https://github.com/zalandoresearch/fashion-mnist
Split used: 55,000 train / 5,000 validation / 10,000 test (fixed seed). Normalisation statistics are computed on the training split only. The dataset downloads automatically on first run.

## Repository contents
| File | Description |
|---|---|
| `kd_fashion_mnist.py` | Complete pipeline: EDA, models, training, distillation, experiments, plots |
| `figures/` | Generated plots (after running) |
| `results_table.csv`, `results_all_runs.csv`, `results_temperature.csv`, `results_alpha.csv` | Generated results (after running) |

## How to run

### Option A: Google Colab (easiest)
1. Open https://colab.research.google.com and create a new notebook.
2. Upload `kd_fashion_mnist.py` using the Files panel on the left.
3. In a cell, run:
   ```
   %run kd_fashion_mnist.py
   ```
4. Download `figures/` and the `results_*.csv` files before the session ends.

### Option B: Local machine
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
python kd_fashion_mnist.py
```

## Configuration
Set near the experiments section of the script:
- `EPOCHS_T`, `EPOCHS_S`: epochs for teacher and students (default 15)
- `SEEDS`: random seeds for the student runs (default `[0, 1, 2]`)
- `T_MAIN`, `ALPHA_MAIN`: temperature and loss weight used in the main comparison (default 4.0 and 0.7)

If CPU time is a problem, reduce epochs or seeds and state this in the report.

## Reproducibility notes
- Seed 42 for the data split and teacher; seeds 0, 1, 2 for students.
- Model selection uses the validation set only. The test set is used for final reporting.
- Optimizer: Adam, lr 2e-3 with cosine schedule, batch size 128, no data augmentation.
- Hardware and software versions used for the reported results: *(fill in: CPU/RAM or "Google Colab CPU runtime", Python, torch, numpy, pandas, scikit-learn versions)*

## Results
*(Fill in after running: paste the contents of `results_table.csv` and reference the figures.)*

## Authors
*(Add both team members' names and roll numbers.)*

## References
- Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network.
- Xiao, Rasul, Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
