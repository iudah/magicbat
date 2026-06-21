
#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void softmax_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent = self->parent[0];

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    auto tmp =
        tensor_softmax_backward((Tensor)parent, self->grad, *(u32 *)self->ctx);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent->base.ndims, parent->base.shape);
    tensor_add_inplace(parent->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

Tensor var_softmax(Tensor var, i32 axis) {

  axis = axis < 0 ? var->ndims + axis : axis;

  Tensor tmp = tensor_softmax(var, axis);
  if (!tmp)
    return nullptr;

  if (var->is_tensor_type || !var->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  u32 *ctx = tmalloc(sizeof(*ctx));
  *ctx = axis;
  var_track_parent(res, var, nullptr, (VarOp){softmax_backward_fn, nullptr},
                   ctx);

  return res;
}
