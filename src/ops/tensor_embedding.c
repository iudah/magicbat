#include "tensor.h"
#include "tensor_prot.h"
#include <string.h>

Tensor tensor_embedding(Tensor input_tokens, Tensor weight_matrix) {

  auto batch = input_tokens->shape[0];
  auto seq = input_tokens->shape[1];
  auto d_model = weight_matrix->shape[1];

  auto embedding = tensor_new(3, (u32[]){batch, seq, d_model});

  for (u32 i = 0; i < batch; ++i) {
    for (u32 j = 0; j < seq; ++j) {
      auto weight_idx = (u32)input_tokens->data->data[(i * seq) + j];
      memcpy(&embedding->data->data[(i * seq + j) * d_model],
             &weight_matrix->data->data[weight_idx * d_model],
             sizeof(f32) * d_model);
    }
  }
  return embedding;
}

Tensor tensor_embedding_backward(Tensor input_tokens, Tensor weight_matrix,
                                 Tensor grad) {

  auto batch = input_tokens->shape[0];
  auto seq = input_tokens->shape[1];
  auto d_model = weight_matrix->shape[1];

  auto weight_grad = tensor_new(weight_matrix->ndims, weight_matrix->shape);

  for (u32 i = 0; i < batch; ++i) {
    for (u32 j = 0; j < seq; ++j) {
      auto weight_idx = (u32)input_tokens->data->data[(i * seq) + j];
      auto w_grad = &weight_grad->data->data[weight_idx * d_model];
      auto e_grad = &grad->data->data[(i * seq + j) * d_model];

      for (u32 k = 0; k < d_model; ++k) {
        w_grad[k] += e_grad[k];
      }
    }
  }
  return weight_grad;
}
