#include "tensor.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void sum_all_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent = (Var)self->parent[0];

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    tensor_add_scalar_inplace(parent->grad, self->grad->data->data[0]);
  }
}

Tensor var_sum(Tensor var) {

  Tensor tmp = tensor_scalar(tensor_sum(var));
  if (!tmp)
    return nullptr;

  if (var->is_tensor_type || !var->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, var, nullptr, (VarOp){sum_all_backward_fn, nullptr},
                   nullptr);

  return res;
}
