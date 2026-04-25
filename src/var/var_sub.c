#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void sub_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = self->parent[0];
  Var b = self->parent[1];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    tensor_add_inplace(a->grad, self->grad);
  }
  if (b && !b->base.is_tensor_type && b->base.requires_grad) {
    if (!b->grad) {
      b->grad = tensor_zero(b->base.ndims, b->base.shape);
    }
    tensor_sub_inplace(b->grad, self->grad);
  }
}

Tensor var_sub(Tensor a, Tensor b) {

  Tensor tmp = tensor_sub(a, b);
  if (!tmp)
    return NULL;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){sub_backward_fn, NULL}, NULL);

  return res;
}
