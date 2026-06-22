#include "sgd.h"
#include "tensor.h"
#include "test_mha.h"
#include "type_alias.h"
#include "var.h"
#include <math.h>
#include <stdio.h>

#define BATCH_SIZE 1
#define SEQ_LEN 32
#define NHEADS 2
#define HEAD_DIM 8
#define D_MODEL HEAD_DIM *NHEADS
#define LR 0.01
#define EPOCH 10

void test_discrete_mha(u32 seq_len) {
  u32 batch = BATCH_SIZE;
  u32 d_model = D_MODEL;
  u32 n_heads = NHEADS;

  f32 sqrt_d = sqrtf((f32)(d_model / n_heads));

  // Initialize Vars (Use same random seeds/values as PyTorch)
  auto input = tensor_new(3, (u32[]){batch, seq_len, d_model});
  auto w_q = track_replace_untracked(tensor_new(2, (u32[]){d_model, d_model}));
  auto w_k = track_replace_untracked(tensor_new(2, (u32[]){d_model, d_model}));
  auto w_v = track_replace_untracked(tensor_new(2, (u32[]){d_model, d_model}));
  auto target_tensor = tensor_new(3, (u32[]){batch, seq_len, d_model});
  SgdOptimizer optim = sgd_optimizer_new((Tensor[]){w_q, w_k, w_v}, 3, LR);

  for (u32 i = 0; i < input->data->nelements; ++i) {
    input->data->data[i] = X[i];
  }
  for (u32 i = 0; i < w_q->data->nelements; ++i) {
    w_q->data->data[i] = W_qkv[i];
  }
  auto w_key = W_qkv + w_q->data->nelements;
  for (u32 i = 0; i < w_k->data->nelements; ++i) {
    w_k->data->data[i] = w_key[i];
  }
  auto w_value = w_key + w_k->data->nelements;
  for (u32 i = 0; i < w_v->data->nelements; ++i) {
    w_v->data->data[i] = w_value[i];
  }
  for (u32 i = 0; i < target_tensor->data->nelements; ++i) {
    target_tensor->data->data[i] = target[i];
  }

  for (u32 epoch = 0; epoch < EPOCH; ++epoch) {
    // 1. Fused Projection & Reshape
    auto query = var_bmm(input, w_q);
    auto key = var_bmm(input, w_k);
    auto value = var_bmm(input, w_v);

    // 2. Attention Math
    auto scores = var_bmm_transpose_b(query, key);
    auto scaled = var_scale(scores, 1 / sqrt_d);
    auto attn = var_softmax(scaled, -1);
    auto out = var_bmm(attn, value);

    // 3. Loss & Backward
    auto loss = var_mse_loss(out, target_tensor);
    var_backward(loss);

    printf("Discrete Epoch %u | Loss: %f\n", epoch, loss->data->data[0]);

    // 4. SGD Step
    sgd_optimize(optim,
                 (Tensor[]){var_grad(w_q), var_grad(w_k), var_grad(w_v)});

    // 5. Zero Grads & GC
    var_zero_grad(loss);
  }
}
