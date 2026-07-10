import torch
import torch.nn.functional as F
import numpy as np
import sys

# 1. Setup deterministic environment
rng = np.random.default_rng(seed=42)
batch_size, vocab_size = 2, 5

# 2. Logits (Requires Grad!)
logits_np = rng.standard_normal((batch_size, vocab_size)).astype(np.float32)
logits = torch.tensor(logits_np, requires_grad=True)

# 3. Create Targets for the 3 Paths
# A. Indexed
targets_indexed_np = np.array([3, 1], dtype=np.int64)
targets_indexed = torch.tensor(targets_indexed_np)

# B. One-Hot
targets_one_hot_np = np.zeros((batch_size, vocab_size), dtype=np.float32)
targets_one_hot_np[0, 3] = 1.0
targets_one_hot_np[1, 1] = 1.0
targets_one_hot = torch.tensor(targets_one_hot_np)

# C. Probs
targets_probs_np = rng.random((batch_size, vocab_size)).astype(np.float32)
targets_probs_np = targets_probs_np / targets_probs_np.sum(axis=-1, keepdims=True)
targets_probs = torch.tensor(targets_probs_np)

# 4. Incoming Gradient (dLoss)
dLoss_np = rng.standard_normal((batch_size,)).astype(np.float32)
dLoss = torch.tensor(dLoss_np)

# 5. Forward and Backward Passes
# A. Indexed
loss_indexed = F.cross_entropy(logits, targets_indexed, reduction='none')
loss_indexed.backward(dLoss, retain_graph=True)
dX_indexed = logits.grad.clone()
logits.grad.zero_()

# B. One-Hot
loss_one_hot = F.cross_entropy(logits, targets_one_hot, reduction='none')
loss_one_hot.backward(dLoss, retain_graph=True)
dX_one_hot = logits.grad.clone()
logits.grad.zero_()

# C. Probs
loss_probs = F.cross_entropy(logits, targets_probs, reduction='none')
loss_probs.backward(dLoss, retain_graph=True)
dX_probs = logits.grad.clone()
logits.grad.zero_()

# 6. Export to C Header
with open("test_crossentropy.h", "w") as f:
    outputs = [
        f"f32 {name}[] = {{ { np.array2string(param.detach().numpy().reshape(-1), separator=', ',threshold=sys.maxsize)[1:-1] } }};" 
        for name, param in [
            ("logits_np", logits), ("dLoss_np", dLoss),
            ("targets_one_hot_np", targets_one_hot), ("targets_probs_np", targets_probs),
            ("loss_indexed_target", loss_indexed), ("loss_one_hot_target", loss_one_hot), ("loss_probs_target", loss_probs),
            ("dX_indexed_target", dX_indexed), ("dX_one_hot_target", dX_one_hot), ("dX_probs_target", dX_probs)
        ]
    ]
    # Indexed targets are unsigned ints
    outputs.append(f"u32 targets_indexed_np[] = {{ { np.array2string(targets_indexed_np, separator=', ',threshold=sys.maxsize)[1:-1] } }};")

    f.write(f"#ifndef TEST_CROSSENTROPY_H\n#define TEST_CROSSENTROPY_H\n#include \"type_alias.h\"\n#define VOCAB_SIZE {vocab_size}\n#define BATCH_SIZE {batch_size}\n")
    for line in outputs:
        f.write(line + "\n")
    f.write("#endif\n")

print("Generated test_crossentropy.h successfully!")
