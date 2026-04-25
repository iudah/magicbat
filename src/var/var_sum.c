#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void sum_all_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = (Var)self->parent[0];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    tensor_add_scalar_inplace(a->grad, self->grad->data->data[0]);
  }
}

Tensor var_sum_all(Tensor t) {

  Tensor tmp = tensor_scalar(tensor_sum_all(t));
  if (!tmp)
    return NULL;

  if (t->is_tensor_type || !t->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, t, NULL, (VarOp){sum_all_backward_fn, NULL}, NULL);

  return res;
}
