#include "sgd.h"
#include "tensor.h"
#include "test_mha.h"
#include "type_alias.h"
#include "var.h"
#include <stdio.h>

#define BATCH_SIZE 1
#define SEQ_LEN 32
#define NHEADS 2
#define HEAD_DIM 8
#define D_MODEL HEAD_DIM *NHEADS
#define LR 0.01
#define EPOCH 10

void test_fused_mha(u32 seq_len) {
  u32 batch = BATCH_SIZE;
  u32 d_model = D_MODEL;
  u32 n_heads = NHEADS;

  // Use IDENTICAL starting values as the discrete test
  auto input = tensor_new(3, (u32[]){batch, seq_len, d_model});
  auto w_qkv =
      track_replace_untracked(tensor_new(2, (u32[]){d_model, 3 * d_model}));
  Tensor target_tensor = tensor_new(3, (u32[]){batch, seq_len, d_model});
  SgdOptimizer optim = sgd_optimizer_new(&w_qkv, 1, LR);

  for (u32 i = 0; i < input->data->nelements; ++i) {
    input->data->data[i] = X[i];
  }
  for (u32 i = 0; i < w_qkv->data->nelements; ++i) {
    w_qkv->data->data[i] = W_qkv[i];
  }
  for (u32 i = 0; i < target_tensor->data->nelements; ++i) {
    target_tensor->data->data[i] = target[i];
  }

  for (u32 epoch = 0; epoch < EPOCH; ++epoch) {
    // 1. Monolithic Forward Pass
    // mha_d is passed as (d_model / n_heads) based on your implementation
    auto out = var_multihead_attention(input, w_qkv, n_heads,
                                       (f32)(d_model / n_heads));

    // 2. Loss & Backward
    Tensor loss = var_mse_loss(out, target_tensor);
    var_backward(
        loss); // This triggers your massive multihead_attention_backward_fn

    printf("Fused Epoch %u | Loss: %f\n", epoch, loss->data->data[0]);

    // 3. SGD Step
    sgd_optimize(optim, (Tensor[]){var_grad(w_qkv)});

    // 4. Zero Grads & GC
    var_zero_grad(loss);
  }
}
