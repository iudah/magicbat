#include "tensor.h"
#include "var_prot.h"
#include <stdint.h>

void copy_data_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var var_a = (Var)self->parent[0];

  if (var_a && !var_a->base.is_tensor_type && var_a->base.requires_grad) {
    if (!var_a->grad) {
      var_a->grad = tensor_zero(var_a->base.ndims, var_a->base.shape);
    }
    auto tmp =
        tensor_sum_to_shape(self->grad, var_a->base.ndims, var_a->base.shape);
    tensor_add_inplace(var_a->grad, tmp);
    var_destroy(tmp);
  }
}

bool var_copy_data(Tensor dst, const Tensor src) {
  auto var = (Var)dst;
  if (!dst->is_tensor_type && dst->requires_grad &&
      !(var->op.backward == nullptr ||
        var->op.backward == copy_data_backward_fn)) {
    return false;
  }
  bool copied = tensor_copy_data(dst, src);

  if (!copied)
    return copied;

  if ((dst->is_tensor_type || !dst->requires_grad) &&
      (src->is_tensor_type || !src->requires_grad)) {
    dst->requires_grad = false;
    return copied;
  }

  var_track_parent(dst, src, nullptr, (VarOp){copy_data_backward_fn, nullptr},
                   nullptr);
  return copied;
}
