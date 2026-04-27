# Multilayer Perceptron — Breast Cancer Classification

A Multilayer Perceptron (MLP) implemented from scratch in Python/NumPy to classify
breast cancer tumors as **malignant (M)** or **benign (B)** using the Wisconsin
Breast Cancer dataset.

No machine learning library is used for the network itself — only NumPy for matrix
operations and Matplotlib for visualization.

---

## Core Concepts

### Feedforward

The feedforward pass is how the network produces a prediction from an input.
Data flows **forward** through each layer, one by one:

1. Each layer computes a weighted sum of its inputs: `Z = X · W + b`
2. An activation function is applied to introduce non-linearity: `A = f(Z)`
3. The output of one layer becomes the input of the next.
4. The final layer uses **softmax**, which converts raw scores into probabilities that sum to 1.

The result is a vector `[P(Benign), P(Malignant)]` for each sample.

### Backpropagation

Backpropagation is the algorithm used to compute how much each weight contributed
to the error, so we can adjust them in the right direction.

It applies the **chain rule** of calculus layer by layer, starting from the output
and moving backward to the input:

1. Compute the error at the output: for softmax + cross-entropy, the gradient simplifies to `delta = A - Y` (where `Y` is the one-hot encoded label).
2. For each layer in reverse order:
   - Compute the gradient of the weights: `dW = input.T · delta / n`
   - Compute the gradient of the biases: `db = mean(delta)`
   - Propagate the error backward: `delta_prev = delta · W.T` (multiplied by the activation derivative for hidden layers)
3. Each weight now has a gradient that tells us how much to change it.

### Gradient Descent

Gradient descent is the optimization algorithm that uses the gradients from
backpropagation to update the weights and reduce the loss.

**Plain SGD** (mini-batch):
```
W = W - lr * dW
b = b - lr * db
```

**SGD with Momentum** (bonus — accumulates past gradients to accelerate convergence):
```
v = 0.9 * v - lr * dW
W = W + v
```

The velocity `v` acts as memory of past gradients.
With `momentum = 0.9`, 90% of the previous velocity is kept at each step,
making convergence faster and smoother than plain SGD.

At each epoch, the training data is shuffled and divided into small **mini-batches**
(e.g. 8–32 samples). One full pass over all mini-batches = one epoch.
The learning rate `lr` controls the size of each update step.

### Z-score normalization

Each feature is scaled to mean ≈ 0 and std ≈ 1:
```
x_normalized = (x - mean) / std
```
Without this, features with large values (`area_mean` ≈ 654) would dominate
features with small values (`fractal_dimension_mean` ≈ 0.06).
The normalization parameters (mean, std) are computed on the training set and
reused at prediction time.

### Learning curves

Two graphs are displayed after training:
1. **Loss** — train vs validation binary cross-entropy per epoch
2. **Accuracy** — train vs validation accuracy per epoch

Used to detect **overfitting** (val loss rises while train loss falls) or
**underfitting** (both losses stay high).

---

## Project Overview

The dataset (`data/data.csv`) contains 569 samples with 32 columns:
- Column `id`: excluded (not a feature)
- Column `diagnosis`: the label to predict (`M` or `B`)
- 30 remaining columns: numeric features describing cell nucleus characteristics
  (radius, texture, perimeter, area, smoothness, etc.)

### Programs

| Program | Role |
|---|---|
| `src/prepare_data.py` | **Step 1** — preprocess and split the dataset into train / validation |
| `src/train.py` | **Step 2** — train the MLP (mandatory) |
| `src/predict.py` | **Step 3** — load a trained model and evaluate it |
| `src/train_bonus.py` | **Bonus** — momentum optimizer, early stopping, extended metrics, comparison |

---

## Setup

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

---

## Mandatory workflow

### Step 1 — Prepare the data

Loads the raw CSV, normalizes the 30 features with z-score, and splits into
train (80%) and validation (20%).

```bash
python3 -m src.prepare_data data/data.csv --target diagnosis
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--val-ratio` | `0.2` | Fraction of data used for validation |
| `--seed` | `42` | Random seed for reproducibility |
| `--save-dir` | `data/processed` | Output directory |

**Output files:**
- `data/processed/dataset_splits.npz` — train/val arrays
- `data/processed/preprocessing_metadata.json` — feature names, means, stds

---

### Step 2 — Train the model

Loads the preprocessed arrays, builds the MLP, trains with mini-batch gradient
descent and backpropagation, then saves the model and displays learning curves.

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
| `--activation` | `sigmoid` | Hidden activation (`sigmoid`, `relu`, `tanh`) |
| `--epochs` | `100` | Number of training epochs |
| `--loss` | `categoricalCrossentropy` | Loss function |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.01` | Gradient descent step size |
| `--seed` | `42` | Random seed for reproducibility |
| `--save` | `saved_model.npy` | Path to save the model |
| `--data` | `data/processed/dataset_splits.npz` | Path to preprocessed data |
| `--config` | `None` | Optional JSON config file |

**Custom architecture (matching the subject example):**
```bash
python3 -m src.train --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
    --batch_size 8 --learning_rate 0.0314
```

**Using a JSON config file:**
```bash
python3 -m src.train --config network_config.json
```

Config file format (all keys optional, CLI args take priority):
```json
{
  "layer": [24, 24],
  "activation": "sigmoid",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.01
}
```

**Output files:**
- `saved_model.npy` — weights, biases, and architecture
- `learning_curves.png` — loss and accuracy graphs

---

### Step 3 — Predict and evaluate

Loads the saved model, normalizes the input CSV with the same parameters used
at training time, runs inference, and evaluates with the binary cross-entropy
formula:

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

### Full mandatory workflow (3 commands)

```bash
# Activate virtual environment
source myenv/bin/activate

# 1. Prepare the data (split + normalize)
python3 -m src.prepare_data data/data.csv --target diagnosis

# 2. Train the model
python3 -m src.train --layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314

# 3. Predict and evaluate
python3 -m src.predict data/data.csv
```

---

## Bonus features (`src/train_bonus.py`)

The 5 bonus features are all grouped in a single program.

### The 5 bonuses

| # | Feature | How to activate |
|---|---|---|
| 1 | **SGD with Momentum** optimizer | `--optimizer momentum` (default) |
| 2 | **Early stopping** | `--early_stopping --patience N` |
| 3 | **Extended metrics** (precision, recall, F1, confusion matrix) | `--predict data/data.csv` |
| 4 | **Metrics history** summary table | displayed automatically |
| 5 | **Multiple curves** on the same graph (SGD vs Momentum) | `--compare` |

### Bonus 1 — SGD with Momentum

Plain SGD updates weights directly from the current gradient:
```
W = W - lr * dW
```
SGD with Momentum adds a velocity term that accumulates past gradients:
```
v = 0.9 * v - lr * dW
W = W + v
```
The velocity acts like a ball rolling downhill: it accelerates in consistent
directions and dampens oscillations when the gradient keeps changing sign.

### Commands

**Default run (Momentum + no early stopping):**
```bash
python3 -m src.train_bonus
```

**With early stopping:**
```bash
python3 -m src.train_bonus --early_stopping --patience 15
```

**Full run — train + extended evaluation:**
```bash
python3 -m src.train_bonus --early_stopping --patience 15 --predict data/data.csv
```

**Example output (train + early stopping + metrics):**
```
x_train shape : (456, 30)
x_valid shape : (113, 30)

Optimizer         : momentum  (lr=0.01)
Early stopping    : patience=15, monitor=val_loss

epoch 01/100 - loss: 0.6684 - val_loss: 0.6146 - acc: 0.6184 - val_acc: 0.6637
...
Early stopping triggered at epoch 42. Restoring weights from epoch 32 (val_loss=0.0891).
> saving model 'saved_model.npy' to disk...

--- Metrics history summary ---
 Epoch  train_loss    val_loss   train_acc     val_acc
------------------------------------------------------
     1      0.6684      0.6146      0.6184      0.6637
    32      0.0712      0.0891      0.9825      0.9735 <- best val_loss
    42      0.0601      0.0963      0.9868      0.9735

─────────────────────────────────────────────
Binary cross-entropy loss : 0.0780
Accuracy                  : 0.9807  (558/569)
─────────────────────────────────────────────

     Class   Precision    Recall        F1   Support
----------------------------------------------------
         B      0.9833    0.9888    0.9860       357
         M      0.9810    0.9717    0.9763       212
----------------------------------------------------
  accuracy                          0.9807       569
 macro avg                          0.9792       569

Confusion Matrix:
               B       M  (predicted)
-------------------------------------
       B     353       4
       M       6     206
(actual)
```

**Compare SGD vs Momentum on the same graph:**
```bash
python3 -m src.train_bonus --compare --epochs 50 --batch_size 8
```

Produces `comparison.png` with both training runs overlaid.

**Available arguments:**

| Argument | Default | Description |
|---|---|---|
| `--layer N [N ...]` | `24 24` | Hidden layer sizes |
| `--activation` | `sigmoid` | Hidden activation |
| `--epochs` | `100` | Max number of epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--learning_rate` | `0.01` | Step size |
| `--optimizer` | `momentum` | `sgd` or `momentum` |
| `--momentum` | `0.9` | Momentum coefficient |
| `--early_stopping` | `False` | Enable early stopping |
| `--patience` | `10` | Epochs without improvement before stopping |
| `--monitor` | `val_loss` | `val_loss` or `val_accuracy` |
| `--predict` | `None` | CSV to evaluate after training |
| `--compare` | `False` | Train SGD vs Momentum and overlay curves |
| `--save` | `saved_model.npy` | Path to save the model |

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
│   ├── prepare_data.py                   # Mandatory step 1: split and normalize
│   ├── train.py                          # Mandatory step 2: train (plain SGD)
│   ├── predict.py                        # Mandatory step 3: predict and evaluate
│   ├── train_bonus.py                    # Bonus: momentum, early stopping, metrics, comparison
│   ├── model.py                          # Network container + training loop + plot_histories()
│   ├── layers.py                         # DenseLayer (forward + backward)
│   ├── activations.py                    # Sigmoid, softmax, relu, tanh
│   ├── optimizers.py                     # SGD and SGDMomentum
│   ├── metrics.py                        # Precision, recall, F1, confusion matrix
│   └── callbacks.py                      # EarlyStopping
├── utils/
│   ├── csv_utils.py                      # CSV parsing utilities
│   ├── datasets.py                       # Breast cancer column schema
│   ├── data_split.py                     # Train/val splitting
│   └── preprocessing.py                  # Z-score normalization + feature selection
├── saved_model.npy                       # Trained model (generated)
├── learning_curves.png                   # Training graphs (generated)
├── comparison.png                        # SGD vs Momentum graph (generated by --compare)
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
  [Dense 2, softmax]     <- output: [P(Benign), P(Malignant)]
```

- **Activation (hidden):** sigmoid — squashes values to (0, 1)
- **Activation (output):** softmax — produces a probability distribution
- **Loss:** binary cross-entropy
- **Optimizer:** mini-batch SGD (mandatory) / SGD with Momentum (bonus)
- **Weight init:** He Uniform

