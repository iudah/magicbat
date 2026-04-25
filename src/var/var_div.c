#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void div_backward_fn(Var self_) {
  Var self = (Var)self_;

  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = self->parent[0];
  Var b = self->parent[1];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    auto tmp = tensor_div(self->grad, (Tensor)b);
    auto tmp_red = tensor_sum_to_shape(tmp, a->base.ndims, a->base.shape);
    tensor_add_inplace(a->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
  if (b && !b->base.is_tensor_type && b->base.requires_grad) {
    if (!b->grad) {
      b->grad = tensor_zero(b->base.ndims, b->base.shape);
    }
    auto tmp = tensor_mul((Tensor)a, self->grad);
    auto db = tensor_divisor_backward((Tensor)b, tmp);
    auto db_red = tensor_sum_to_shape(db, b->base.ndims, b->base.shape);
    tensor_add_inplace(b->grad, db_red);
    var_destroy(db_red);
    var_destroy(tmp);
    var_destroy(db);
  }
}

Tensor var_div(Tensor a, Tensor b) {

  Tensor tmp = tensor_mul(a, b);
  if (!tmp)
    return NULL;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){div_backward_fn, NULL}, NULL);

  return res;
}
