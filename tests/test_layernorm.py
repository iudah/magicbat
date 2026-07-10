import torch
import torch.nn.functional as F
import numpy as np
import sys

# 1. Setup deterministic environment
rng = np.random.default_rng(seed=42)

batch_size = 2
seq_len = 8
d_model = 16

# 2. Initialize Tensors
X_np = rng.standard_normal((batch_size, seq_len, d_model))
grad_np = rng.standard_normal((batch_size, seq_len, d_model))

X = torch.tensor(X_np, requires_grad=True)
dy = torch.tensor(grad_np)

# 3. Forward Pass (No Gamma/Beta weights to match your current C Engine)
out = F.layer_norm(X, [d_model])

# 4. Backward Pass
out.backward(dy)

# 5. Export to C Header
with open("test_layernorm.h", "w") as f:
    outputs = [
        f"f32 {name}[] = {{ { np.array2string(param.detach().numpy().reshape(-1), separator=', ',threshold=sys.maxsize)[1:-1] } }};" 
        for name, param in [("X", X), ("dy", dy), ("out_target", out), ("dX_target", X.grad)]
    ]

    f.write("#ifndef TEST_LAYERNORM_H\n#define TEST_LAYERNORM_H\n#include \"type_alias.h\"\n")
    for line in outputs:
        f.write(line + "\n")
    f.write("#endif\n")

print("--- PyTorch LayerNorm Test ---")
print(f"Sample Forward Out [0,0,:4]: {out[0,0,:4].detach().numpy()}")
print(f"Sample Backward dX [0,0,:4]: {X.grad[0,0,:4].detach().numpy()}")
