#include "tensor.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void emb_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var weight_matrix = self->parent[1];

  if (weight_matrix && !weight_matrix->base.is_tensor_type &&
      weight_matrix->base.requires_grad) {
    if (weight_matrix->grad) {
      var_destroy(weight_matrix->grad);
    }
    weight_matrix->grad = tensor_embedding_backward(
        &self->parent[0]->base, &weight_matrix->base, self->grad);
  }
}

Tensor var_embedding(Tensor input_tokens, Tensor weight_matrix) {

  Tensor tmp = tensor_embedding(input_tokens, weight_matrix);
  if (!tmp)
    return nullptr;

  if (weight_matrix->is_tensor_type || !weight_matrix->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, input_tokens, weight_matrix,
                   (VarOp){emb_backward_fn, nullptr}, nullptr);

  return res;
}
