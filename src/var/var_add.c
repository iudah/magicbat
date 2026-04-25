#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void add_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = (Var)self->parent[0];
  Var b = (Var)self->parent[1];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    auto tmp = tensor_sum_to_shape(self->grad, a->base.ndims, a->base.shape);
    tensor_add_inplace(a->grad, tmp);
    var_destroy(tmp);
  }
  if (b && !b->base.is_tensor_type && b->base.requires_grad) {
    if (!b->grad) {
      b->grad = tensor_zero(b->base.ndims, b->base.shape);
    }
    auto tmp = tensor_sum_to_shape(self->grad, b->base.ndims, b->base.shape);
    tensor_add_inplace(b->grad, tmp);
    var_destroy(tmp);
  }
}

Tensor var_add(Tensor a, Tensor b) {

  Tensor tmp = tensor_add(a, b);
  if (!tmp)
    return NULL;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){add_backward_fn, NULL}, NULL);

  return res;
}
