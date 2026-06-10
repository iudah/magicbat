#include "tensor.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void relu_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent = self->parent[0];

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    auto tmp = tensor_relu_backward((Tensor)parent, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent->base.ndims, parent->base.shape);
    tensor_add_inplace(parent->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

Tensor var_relu(Tensor var) {

  Tensor tmp = tensor_relu(var);
  if (!tmp)
    return nullptr;

  if (var->is_tensor_type || !var->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, var, nullptr, (VarOp){relu_backward_fn, nullptr},
                   nullptr);

  return res;
}
