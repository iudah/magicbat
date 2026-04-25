#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void matmul_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = self->parent[0];
  Var b = self->parent[1];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    auto b_T = tensor_transpose((Tensor)b);
    auto da = tensor_matmul(self->grad, b_T);
    tensor_add_inplace(a->grad, da);
    var_destroy(b_T);
    var_destroy(da);
  }
  if (b && !b->base.is_tensor_type && b->base.requires_grad) {
    if (!b->grad) {
      b->grad = tensor_zero(b->base.ndims, b->base.shape);
    }
    auto a_T = tensor_transpose((Tensor)a);
    auto db = tensor_matmul(a_T, self->grad);
    tensor_add_inplace(b->grad, db);
    var_destroy(a_T);
    var_destroy(db);
  }
}

Tensor var_matmul(Tensor a, Tensor b) {

  Tensor tmp = tensor_matmul(a, b);
  if (!tmp)
    return NULL;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){matmul_backward_fn, NULL}, NULL);

  return res;
}
