# Multilayer Perceptron — Breast Cancer Classification

A Multilayer Perceptron (MLP) implemented from scratch in Python/NumPy to classify breast cancer tumors as **malignant (M)** or **benign (B)** using the Wisconsin Breast Cancer dataset.

No machine learning library is used for the network itself — only NumPy for matrix operations and Matplotlib for visualization.

---

## Project Overview

The dataset (`data/data.csv`) contains 569 samples with 32 columns:
- Column `id`: excluded (not a feature)
- Column `diagnosis`: the label to predict (`M` or `B`)
- 30 remaining columns: numeric features describing cell nucleus characteristics (radius, texture, perimeter, area, smoothness, etc.)

The full pipeline is split into three programs:

| Program | Description |
|---|---|
| `src/prepare_data.py` | Preprocess and split the dataset into train / validation |
| `src/train.py` | Train the MLP using backpropagation + gradient descent |
| `src/predict.py` | Load a trained model and evaluate it on a dataset |

---

## Setup

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Prepare the data

This program:
- Loads the raw CSV
- Selects the 30 numeric features (excludes `id` and `diagnosis`)
- Applies **z-score normalization**: `x = (x - mean) / std`
- Splits the dataset into **train (80%)** and **validation (20%)**
- Saves the processed arrays and normalization parameters

```bash
python3 -m src.prepare_data data/data.csv --target diagnosis
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--val-ratio` | `0.2` | Fraction of data used for validation |
| `--seed` | `42` | Random seed for reproducibility |
| `--save-dir` | `data/processed` | Output directory |

**Example:**
```bash
python3 -m src.prepare_data data/data.csv --target diagnosis --val-ratio 0.2 --seed 42
```

**Output files:**
- `data/processed/dataset_splits.npz` — train/val arrays
- `data/processed/preprocessing_metadata.json` — feature names, means, stds

---

## Step 2 — Train the model

This program:
- Loads the preprocessed arrays from Step 1
- Builds the MLP with the specified architecture
- Trains using **mini-batch gradient descent** and **backpropagation**
- Displays loss and accuracy at every epoch
- Saves the trained model (weights + architecture) to `saved_model.npy`
- Displays two learning curve graphs (loss and accuracy)

```bash
python3 -m src.train
```

**Example output:**
```
x_train shape : (456, 30)
x_valid shape : (113, 30)

epoch 01/100 - loss: 0.5487 - val_loss: 0.5296 - acc: 0.6689 - val_acc: 0.6903
epoch 02/100 - loss: 0.4602 - val_loss: 0.4587 - acc: 0.9101 - val_acc: 0.9292
...
epoch 100/100 - loss: 0.0580 - val_loss: 0.1020 - acc: 0.9868 - val_acc: 0.9646
> saving model 'saved_model.npy' to disk...
```

**Available arguments:**

| Argument | Default | Description |
|---|---|---|
| `--layer N [N ...]` | `24 24` | Hidden layer sizes |
| `--activation` | `sigmoid` | Hidden layer activation (`sigmoid`, `relu`, `tanh`) |
| `--epochs` | `100` | Number of training epochs |
| `--loss` | `categoricalCrossentropy` | Loss function |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.01` | Gradient descent step size |
| `--seed` | `42` | Random seed for reproducibility |
| `--save` | `saved_model.npy` | Path to save the model |
| `--data` | `data/processed/dataset_splits.npz` | Path to preprocessed data |
| `--config` | `None` | Optional JSON config file (CLI args take priority) |

**Custom architecture example (matching the subject):**
```bash
python3 -m src.train --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
    --batch_size 8 --learning_rate 0.0314
```

**Using a JSON config file:**
```bash
python3 -m src.train --config network_config.json
```

Config file format:
```json
{
  "layer": [24, 24, 24],
  "activation": "sigmoid",
  "epochs": 84,
  "batch_size": 8,
  "learning_rate": 0.0314
}
```

**Output files:**
- `saved_model.npy` — weights, biases, and architecture
- `learning_curves.png` — loss and accuracy graphs

---

## Step 3 — Predict and evaluate

This program:
- Loads the saved model from `saved_model.npy`
- Normalizes the input CSV using the **same means and stds** from training
- Runs a forward pass to obtain predicted probabilities
- Evaluates performance using the **binary cross-entropy** formula:

$$E = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log p_n + (1 - y_n) \log(1 - p_n) \right]$$

where `p` = P(Malignant | x), `y = 1` if malignant, `y = 0` if benign.

```bash
python3 -m src.predict data/data.csv
```

**Example output:**
```
#    Prediction  Confidence  P(Benign)  P(Malignant)
-----------------------------------------------------------------
     0             M      0.9997     0.0003        0.9997
     1             M      0.9994     0.0006        0.9994
...
────────────────────────────────────────
Binary cross-entropy loss : 0.0723
Accuracy                  : 0.9772  (556/569)
────────────────────────────────────────
```

**Available arguments:**

| Argument | Default | Description |
|---|---|---|
| `dataset` | *(required)* | Path to the CSV file to predict on |
| `--model` | `saved_model.npy` | Path to the saved model file |
| `--metadata` | `data/processed/preprocessing_metadata.json` | Normalization parameters |

---

## Full workflow

```bash
# 1. Activate virtual environment
source myenv/bin/activate

# 2. Prepare the data (split + normalize)
python3 -m src.prepare_data data/data.csv --target diagnosis

# 3. Train the model
python3 -m src.train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

# 4. Predict and evaluate
python3 -m src.predict data/data.csv
```

---

## Project structure

```
Multilayer-perceptron/
├── data/
│   ├── data.csv                          # Raw dataset (569 samples, 32 columns)
│   └── processed/
│       ├── dataset_splits.npz            # Train/val arrays (generated)
│       └── preprocessing_metadata.json   # Normalization parameters (generated)
├── src/
│   ├── prepare_data.py                   # Step 1: split and normalize
│   ├── train.py                          # Step 2: train the MLP
│   ├── predict.py                        # Step 3: predict and evaluate
│   ├── model.py                          # Network container + training loop
│   ├── layers.py                         # DenseLayer (forward + backward)
│   └── activations.py                    # Sigmoid, softmax, relu, tanh
├── utils/
│   ├── csv_utils.py                      # CSV parsing utilities
│   ├── datasets.py                       # Breast cancer column schema
│   ├── data_split.py                     # Train/val splitting
│   └── preprocessing.py                  # Normalization + feature selection
├── saved_model.npy                       # Trained model (generated)
├── learning_curves.png                   # Training graphs (generated)
├── requirements.txt
└── README.md
```

---

## Neural network architecture

```
Input (30 features)
        |
  [Dense 24, sigmoid]    <- hidden layer 1
        |
  [Dense 24, sigmoid]    <- hidden layer 2
        |
  [Dense 2, softmax]     <- output layer: [P(Benign), P(Malignant)]
```

- **Activation (hidden layers):** sigmoid (squashes values to (0, 1))
- **Activation (output layer):** softmax (probability distribution over classes)
- **Loss:** binary cross-entropy (equivalent to categorical cross-entropy for 2 classes)
- **Optimizer:** mini-batch gradient descent with backpropagation
- **Weight init:** He Uniform (calibrated for the chosen activation)

---

## Key concepts

### Z-score normalization
Each feature is scaled to have mean ≈ 0 and std ≈ 1:
```
x_normalized = (x - mean) / std
```
Without this, features with large scales (e.g. `area_mean` ≈ 654) would dominate
those with small scales (e.g. `fractal_dimension_mean` ≈ 0.06), causing slow or
unstable convergence.

### Backpropagation
The chain rule is applied layer by layer in reverse order to compute `dL/dW`
for each weight matrix. The combined gradient of **softmax + cross-entropy** is:
```
delta = A_output - Y_onehot
```
This elegant simplification avoids computing the full softmax Jacobian.

### Mini-batch gradient descent
Instead of computing gradients on the full dataset (slow) or one sample at a time
(noisy), we use small batches (e.g. 8 or 32 samples). This balances stability
and speed, and also acts as a natural regularizer.

### Learning curves
Two graphs are displayed after training:
1. **Loss curve** — train vs validation binary cross-entropy per epoch
2. **Accuracy curve** — train vs validation accuracy per epoch

These help detect **overfitting** (val loss rises while train loss falls)
or **underfitting** (both losses remain high).
