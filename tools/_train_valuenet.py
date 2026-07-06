"""Train a value net from self-play data (tools/_gen_valuenet_data.py output).

Pure-numpy training (no sklearn/torch) so the whole pipeline is dependency-light.
Trains a logistic model and a 1-hidden-layer tanh MLP, evaluates both on a
held-out split (accuracy + log-loss), and saves the better one as an .npz in the
format agent.value_net.ValueNet expects:
  logistic: kind="logistic", w[N_FEATURES]
  mlp:      kind="mlp", W1, b1, W2, b2

The bias feature is already the last column of X, so the linear models learn an
intercept through it (no separate bias term needed for logistic).

Run: python tools/_train_valuenet.py --data artifacts/valuenet_data.npz \
        --out submission/agent/value_net.npz --hidden 16
"""
from __future__ import annotations
import argparse, os
import numpy as np


def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(X, y, l2=1e-3, lr=0.5, epochs=4000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w


def train_mlp(X, y, hidden=16, l2=1e-4, lr=0.3, epochs=6000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 0.3, (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.3, hidden)
    b2 = 0.0
    for _ in range(epochs):
        z1 = X @ W1 + b1
        h = np.tanh(z1)
        z2 = h @ W2 + b2
        p = sigmoid(z2)
        dz2 = (p - y) / n
        gW2 = h.T @ dz2 + l2 * W2
        gb2 = dz2.sum()
        dh = np.outer(dz2, W2) * (1 - h ** 2)
        gW1 = X.T @ dh + l2 * W1
        gb1 = dh.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2
        W1 -= lr * gW1; b1 -= lr * gb1
    return W1, b1, W2, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("artifacts", "valuenet_data.npz"))
    ap.add_argument("--out", default=os.path.join("submission", "agent", "value_net.npz"))
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    print(f"data: {X.shape[0]} states x {X.shape[1]} feats  win-rate={y.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    n_val = int(len(y) * args.val_frac)
    Xv, yv = X[:n_val], y[:n_val]
    Xt, yt = X[n_val:], y[n_val:]

    base = yv.mean()
    base_ll = log_loss(yv, np.full_like(yv, yt.mean()))
    print(f"baseline (predict prior {yt.mean():.3f}): val log-loss={base_ll:.4f}")

    # Logistic
    w = train_logistic(Xt, yt)
    pv = sigmoid(Xv @ w)
    log_acc = float(((pv > 0.5) == (yv > 0.5)).mean())
    print(f"logistic:  val acc={log_acc:.3f}  log-loss={log_loss(yv, pv):.4f}")

    # MLP
    W1, b1, W2, b2 = train_mlp(Xt, yt, hidden=args.hidden, seed=args.seed)
    hv = np.tanh(Xv @ W1 + b1)
    pv_mlp = sigmoid(hv @ W2 + b2)
    mlp_acc = float(((pv_mlp > 0.5) == (yv > 0.5)).mean())
    mlp_ll = log_loss(yv, pv_mlp)
    print(f"mlp(h={args.hidden}): val acc={mlp_acc:.3f}  log-loss={mlp_ll:.4f}")

    # Pick lower val log-loss
    if mlp_ll < log_loss(yv, pv):
        print("-> saving MLP")
        np.savez(args.out, kind="mlp", W1=W1, b1=b1, W2=W2, b2=np.array([b2]))
    else:
        print("-> saving logistic")
        np.savez(args.out, kind="logistic", w=w)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
