import torch
import torch.nn.functional as F
import numpy as  np
import math
import sys

# 1. Setup deterministic environment
rng=np.random.default_rng(seed=10)

batch_size = 1
seq_len = 32  
d_model = 16
n_heads = 2
head_dim = seq_len
d_model = head_dim * n_heads
learning_rate = 0.01

# 2. Initialize Tensors (Requires Grad!)
X_np = rng.standard_normal((batch_size, seq_len, d_model))
W_qkv_np = rng.standard_normal((d_model, 3 * d_model))
target_np = rng.standard_normal((batch_size, seq_len, d_model))

'''
with open("test_mha.h", "w") as f:
    outputs = [
        f"static f32 {name}[] = {{ { np.array2string(param.reshape(-1), separator=', ',threshold=sys.maxsize)[1:-1] } }};" 
        for name, param in [("X",X_np),("W_qkv", W_qkv_np), ("target", target_np)]
    ]

    f.write(f"""
#ifndef TEST_MHA_H
#define TEST_MHA_H
""")
    for line in outputs:
        f.write(line)
    f.write(f"""
#endif""")'''

def run_test(seq_len = 8):
    head_dim = 8
    d_model = head_dim * n_heads
    
    X= torch.tensor(X_np.reshape(-1)[:batch_size*seq_len*d_model].reshape(batch_size,seq_len,d_model))
    W_qkv= torch.tensor(W_qkv_np.reshape(-1)[:d_model*3*d_model].reshape(d_model, 3*d_model))
    target= torch.tensor(target_np.reshape(-1)[:batch_size*seq_len*d_model].reshape(batch_size,seq_len,d_model))
    
    W_qkv.requires_grad=True
    
    print(f"--- PyTorch MHA Test (Seq={seq_len}) ---")
    
    for epoch in range(10):
        # --- FORWARD PASS ---
        # 1. Fused Projection: [1, 8, 16] @ [16, 48] -> [1, 8, 48]
        X_qkv = X @ W_qkv
        
        # 2. Reshape and Transpose to [3, 1, 2, 8, 8]
        X_qkv = X_qkv.view(batch_size, seq_len, 3, n_heads, head_dim)
        q, k, v = X_qkv.unbind(dim=2)
        
        q = q.transpose(1, 2) # [1, 2, 8, 8]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        X_qkv.retain_grad()
        # 3. Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        
        # 4. Output Projection Prep
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 5. Loss calculation (Mean Squared Error)
        loss = F.mse_loss(out, target)
        
        # --- BACKWARD PASS ---
        loss.backward()
        
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
        
        # --- OPTIMIZER (Simple SGD) ---
        with torch.no_grad():
            W_qkv -= learning_rate * W_qkv.grad
            W_qkv.grad.zero_()
    
run_test(32)
run_test(8)
