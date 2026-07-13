#ifndef VAR_H
#define VAR_H
#include "tensor.h"

typedef struct var_struct *Var;

Tensor track(Tensor tensor);
void untrack(Tensor var);
bool var_destroy(Tensor var);
bool var_require_grad(Tensor var);
bool var_is_tensor(Tensor var);
Tensor var_grad(Tensor var);

Tensor var_add(Tensor var_a, Tensor var_b);
Tensor var_sub(Tensor var_a, Tensor var_b);
Tensor var_mul(Tensor var_a, Tensor var_b);
Tensor var_scale(Tensor var_a, f32 scalar);
Tensor var_div(Tensor var_a, Tensor var_b);
Tensor var_matmul(Tensor var_a, Tensor var_b);
Tensor var_bmm(Tensor var_a, Tensor var_b);
Tensor var_bmm_transpose_b(Tensor var_a, Tensor var_b);
bool var_copy_data(Tensor dst, Tensor src);
Tensor var_concat(u32 ntensor, Tensor *tensors, u32 axis);
Tensor var_multihead_attention(Tensor input, Tensor W_qkv, u32 n_head,
                               u32 head_dim);
Tensor var_multihead_attention_causal(Tensor input, Tensor W_qkv, u32 n_head,
                                      u32 head_dim);
Tensor var_mse_loss(Tensor pred, Tensor target);
Tensor var_cross_entropy_one_hot(Tensor logits, Tensor targets, i32 axis);
Tensor var_cross_entropy_probs(Tensor logits, Tensor targets, i32 axis);
Tensor var_cross_entropy_indexed(Tensor logits, Tensor targets, i32 axis);
Tensor var_cross_entropy_loss_one_hot(Tensor logits, Tensor targets, i32 axis);
Tensor var_cross_entropy_loss_probs(Tensor logits, Tensor targets, i32 axis);
Tensor var_cross_entropy_loss_indexed(Tensor logits, Tensor targets, i32 axis);
Tensor var_embedding(Tensor input_tokens, Tensor weight_matrix);

Tensor var_relu(Tensor var);
Tensor var_tanh(Tensor var);
Tensor var_sum_all(Tensor var);
Tensor var_layer_norm_axis(Tensor input, u32 axis);
Tensor var_softmax(Tensor var, i32 axis);

bool var_backward(Tensor top);
bool var_backward_with_grad(Tensor top, Tensor grad);
bool var_backward_verbose(Tensor top, bool user_thread_safety_assured,
                          Tensor grad);
bool var_zero_grad(Tensor top);

static inline Tensor track_replace_untracked(Tensor var) {
  auto tracked = track(var);
  if (!var_is_tensor(var))
    return tracked;
  tensor_destroy(var);
  return tracked;
}

#endif
