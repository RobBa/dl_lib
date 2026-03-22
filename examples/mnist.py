import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python_lib"))

import math
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from dl_lib import Tensor, fromNumpy, toNumpy
from dl_lib.nn import Sequential, FfLayer
from dl_lib.nn.activation import LeakyReLU
from dl_lib.train.loss import CrossEntropyWithSoftmax
from dl_lib.train.optim import RmsProp


# ─── data loading ────────────────────────────────────────────────────────────

def load_mnist():
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data.astype(np.float32) / 255.0  # normalize to [0,1]
    y = mnist.target.astype(np.int32)
    return x, y


def to_one_hot(y, n_classes=10):
    n = len(y)
    one_hot = np.zeros((n, n_classes), dtype=np.float32)
    one_hot[np.arange(n), y] = 1.0
    return one_hot


def make_batches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


# ─── network ─────────────────────────────────────────────────────────────────

def make_net():
    net = Sequential()
    net.append(FfLayer(784, 256))
    net.append(LeakyReLU(0.01))
    net.append(FfLayer(256, 128))
    net.append(LeakyReLU(0.01))
    net.append(FfLayer(128, 10))
    return net


# ─── debugging ────────────────────────────────────────────────────────────────

def print_weight_stats(net, batch_num):
    for i, p in enumerate(net.parameters()):
        p_np = toNumpy(p)
        print(
            f"Param {i}: min={p_np.min():.4f} max={p_np.max():.4f} "
            f"mean={p_np.mean():.4f} std={p_np.std():.4f}"
        )


# ─── training ────────────────────────────────────────────────────────────────

def train_epoch(net, loss_fn, optim, x, y, batch_size=64):
    #print_weight_stats(net, 0)

    total_loss = 0.0
    n_batches = 0
    max_batches = math.ceil(x.shape[0] / batch_size)
    for xb, yb in make_batches(x, y, batch_size):
        xTensor = fromNumpy(xb)
        yTensor = fromNumpy(yb)

        pred = net.forward(xTensor)
        loss = loss_fn(yTensor, pred)
        loss.backward()

        optim.clipGradients(1.0)
        optim.step()
        optim.zeroGrad()

        total_loss += loss.getitem(0)
        n_batches += 1
        if n_batches == 1 or n_batches % 10 == 0:
            print(f"Batch {n_batches} / {max_batches}, loss {loss.getitem(0)}")
        #print_weight_stats(net, n_batches)

    return total_loss / n_batches


def evaluate(net, x, y_int, batch_size=256):
    correct = 0
    total = 0
    for xb, yb in make_batches(x, y_int, batch_size, shuffle=False):
        xTensor = fromNumpy(xb)
        pred = net.forward(xTensor)
        pred_np = toNumpy(pred)

        predicted = np.argmax(pred_np, axis=1)
        
        correct += np.sum(predicted == np.argmax(yb, axis=1))
        total += len(yb)
    return correct / total


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load and split data
    x, y_int = load_mnist()
    y = to_one_hot(y_int)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}")

    # setup
    net = make_net()
    loss_fn = CrossEntropyWithSoftmax()
    optim = RmsProp(net.parameters(), 0.00001, 0.95)  # lr and decay

    # training loop
    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = train_epoch(net, loss_fn, optim, x_train, y_train)
        val_acc = evaluate(net, x_val, y_val)
        print(
            f"Epoch {epoch+1}/{n_epochs} "
            f"loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )
