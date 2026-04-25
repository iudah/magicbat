#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void relu_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var a = self->parent[0];

  if (a && !a->base.is_tensor_type && a->base.requires_grad) {
    if (!a->grad) {
      a->grad = tensor_zero(a->base.ndims, a->base.shape);
    }
    auto tmp = tensor_relu_backward((Tensor)self, self->grad);
    tensor_add_inplace(a->grad, tmp);
    var_destroy(tmp);
  }
}

Tensor var_relu(Tensor a) {

  Tensor tmp = tensor_relu(a);
  if (!tmp)
    return NULL;

  if (a->is_tensor_type || !a->requires_grad)
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, NULL, (VarOp){relu_backward_fn, NULL}, NULL);

  return res;
}
