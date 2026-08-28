#include "var_prot.h"
#include "tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void add_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var var_a = (Var)self->parent[0];
  Var var_b = (Var)self->parent[1];

  if (var_a && !var_a->base.is_tensor_type && var_a->base.requires_grad) {
    if (!var_a->grad) {
      var_a->grad = tensor_zero(var_a->base.ndims, var_a->base.shape);
    }
    auto tmp =
        tensor_sum_to_shape(self->grad, var_a->base.ndims, var_a->base.shape);
    tensor_add_inplace(var_a->grad, tmp);
    var_destroy(tmp);
  }
  if (var_b && !var_b->base.is_tensor_type && var_b->base.requires_grad) {
    if (!var_b->grad) {
      var_b->grad = tensor_zero(var_b->base.ndims, var_b->base.shape);
    }
    auto tmp =
        tensor_sum_to_shape(self->grad, var_b->base.ndims, var_b->base.shape);
    tensor_add_inplace(var_b->grad, tmp);
    var_destroy(tmp);
  }
}

Tensor var_add(Tensor var_a, Tensor var_b) {

  Tensor tmp = tensor_add(var_a, var_b);
  if (!tmp)
    return nullptr;

  if ((var_a->is_tensor_type || !var_a->requires_grad) &&
      (var_b->is_tensor_type || !var_b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, var_a, var_b, (VarOp){add_backward_fn, nullptr},
                   nullptr);

  return res;
}
