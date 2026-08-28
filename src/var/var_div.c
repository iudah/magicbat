#include "var_prot.h"
#include "tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void div_backward_fn(Var self) {

  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent_a = self->parent[0];
  Var parent_b = self->parent[1];

  if (parent_a && !parent_a->base.is_tensor_type &&
      parent_a->base.requires_grad) {
    if (!parent_a->grad) {
      parent_a->grad = tensor_zero(parent_a->base.ndims, parent_a->base.shape);
    }
    auto tmp = tensor_div(self->grad, (Tensor)parent_b);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent_a->base.ndims, parent_a->base.shape);
    tensor_add_inplace(parent_a->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
  if (parent_b && !parent_b->base.is_tensor_type &&
      parent_b->base.requires_grad) {
    if (!parent_b->grad) {
      parent_b->grad = tensor_zero(parent_b->base.ndims, parent_b->base.shape);
    }
    auto tmp = tensor_mul((Tensor)parent_a, self->grad);
    auto grad_wrt_b = tensor_divisor_backward((Tensor)parent_b, tmp);
    auto db_red = tensor_sum_to_shape(grad_wrt_b, parent_b->base.ndims,
                                      parent_b->base.shape);
    tensor_sub_inplace(parent_b->grad, db_red);
    var_destroy(db_red);
    var_destroy(tmp);
    var_destroy(grad_wrt_b);
  }
}

Tensor var_div(Tensor a, Tensor b) {

  Tensor tmp = tensor_div(a, b);
  if (!tmp)
    return nullptr;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){div_backward_fn, nullptr}, nullptr);

  return res;
}
