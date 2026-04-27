import numpy as np

# ----- Model parameters -----
np.random.seed(0)

w1 = np.full((2, 4), 0.025, dtype=np.float32)
b1 = np.full((1, 4), 0.020, dtype=np.float32)
w2 = np.full((4, 1), 0.035, dtype=np.float32)
b2 = np.full((1, 1), 0.030, dtype=np.float32)

lr = 0.1

# XOR dataset
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

def relu(x):
    return np.maximum(0, x)

def relu_backward(x, grad):
    return grad * (x > 0)

for epoch in range(100):
    # ----- Forward -----
    h1 = x @ w1 + b1
    h1_act = relu(h1)
    logits = h1_act @ w2 + b2

    diff = logits - y
    loss = np.mean(diff ** 2)

    print(f"Epoch {epoch}, Loss = {loss:.4f}")

    # ----- Backward -----
    N = x.shape[0]

    grad_logits = 2 * diff / N

    grad_w2 = h1_act.T @ grad_logits
    grad_b2 = np.sum(grad_logits, axis=0, keepdims=True)

    grad_h1_act = grad_logits @ w2.T
    grad_h1 = relu_backward(h1, grad_h1_act)

    grad_w1 = x.T @ grad_h1
    grad_b1 = np.sum(grad_h1, axis=0, keepdims=True)

    # ----- SGD update -----
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2
    