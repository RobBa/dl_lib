import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python_lib"))

import math
import random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from dl_lib import Tensor, Device, fromNumpy, toNumpy
from dl_lib.nn import Sequential, FfLayer
from dl_lib.nn.activation import LeakyReLU
from dl_lib.train.loss import CrossEntropyWithSoftmax
from dl_lib.train.optim import RmsProp

import time

# ─── data loading ────────────────────────────────────────────────────────────

def load_mnist():
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    return x, y


def to_one_hot(y, n_classes=10):
    n = len(y)
    one_hot = np.zeros((n, n_classes), dtype=np.float32)
    one_hot[np.arange(n), y] = 1.0
    return one_hot


def to_gpu(np_arr):
    t = fromNumpy(np_arr)
    t.device = Device.CUDA
    return t


# ─── network ─────────────────────────────────────────────────────────────────

def make_net():
    net = Sequential()
    net.append(FfLayer(784, 256, Device.CUDA))
    net.append(LeakyReLU(0.01))
    net.append(FfLayer(256, 128, Device.CUDA))
    net.append(LeakyReLU(0.01))
    net.append(FfLayer(128, 10, Device.CUDA))
    return net


# ─── training ────────────────────────────────────────────────────────────────

def train_epoch(net, loss_fn, optim, x_gpu, y_gpu, batch_size=64, max_batch_count=50):
    n = x_gpu.dims[0]

    # shuffle both tensors identically on the GPU via a random permutation of row indices
    indices = list(range(n))
    random.shuffle(indices)
    x_shuf = x_gpu.slice(indices)
    y_shuf = y_gpu.slice(indices)

    total_loss = 0.0
    n_batches = 0
    max_batches = math.ceil(n / batch_size)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x_shuf.slice(start, end)
        yb = y_shuf.slice(start, end)

        pred = net.forward(xb)
        loss = loss_fn(yb, pred)
        loss.backward()

        optim.clipGradients(1.0)
        optim.step()
        optim.zeroGrad()

        total_loss += loss.getitem(0)
        n_batches += 1
        #if n_batches == 1 or n_batches % 10 == 0:
        #    print(f"Batch {n_batches} / {max_batches}, loss {loss.getitem(0)}")
        if n_batches == max_batch_count:
          return total_loss / n_batches
        #print_weight_stats(net, n_batches)


def evaluate(net, x_gpu, y_np, batch_size=256):
    n = x_gpu.dims[0]
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred = net.forward(x_gpu.slice(start, end))
        pred.device = Device.CPU
        pred_np = toNumpy(pred)
        predicted = np.argmax(pred_np, axis=1)
        correct += np.sum(predicted == np.argmax(y_np[start:end], axis=1))
    return correct / n


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x, y_int = load_mnist()
    y = to_one_hot(y_int)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}")

    # upload once; all batching and shuffling happens on the GPU
    x_train_gpu = to_gpu(x_train)
    y_train_gpu = to_gpu(y_train)
    x_val_gpu   = to_gpu(x_val)

    net = make_net()
    loss_fn = CrossEntropyWithSoftmax()
    optim = RmsProp(net.parameters(), 0.0001, 0.999)

    # warm up GPU
    BATCH_COUNT = 500
    train_epoch(net, loss_fn, optim, x_train_gpu, y_train_gpu, max_batch_count=BATCH_COUNT)

    times = []
    n_epochs = 5
    for epoch in range(n_epochs):
        
        start = time.perf_counter()
        train_loss = train_epoch(net, loss_fn, optim, x_train_gpu, y_train_gpu, max_batch_count=BATCH_COUNT)
        elapsed = time.perf_counter() - start
        print(f"Elapsed time GPU: {elapsed:.4f}s")

        times.append(elapsed)

    median_time = sorted(times)[len(times) // 2]
    print(f"Median time GPU: {median_time:.4f}s")