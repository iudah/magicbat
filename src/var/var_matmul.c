#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include <stdatomic.h>
#include <stdint.h>

void matmul_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent_a = self->parent[0];
  Var parent_b = self->parent[1];

  if (parent_a && !parent_a->base.is_tensor_type &&
      parent_a->base.requires_grad) {
    if (!parent_a->grad) {
      parent_a->grad = tensor_zero(parent_a->base.ndims, parent_a->base.shape);
    }
    // auto b_T = tensor_transpose((Tensor)parent_b);
    auto grad_wrt_a = tensor_matmul_wrt_a(self->grad, &parent_a->base);
    auto da_red = grad_wrt_a;
    // tensor_sum_to_shape(grad_wrt_a, parent_a->base.ndims,
    // parent_a->base.shape);
    tensor_add_inplace(parent_a->grad, da_red);
    // var_destroy(b_T);
    // var_destroy(da_red);
    var_destroy(grad_wrt_a);
  }
  if (parent_b && !parent_b->base.is_tensor_type &&
      parent_b->base.requires_grad) {
    if (!parent_b->grad) {
      parent_b->grad = tensor_zero(parent_b->base.ndims, parent_b->base.shape);
    }
    auto a_T = tensor_transpose((Tensor)parent_a);
    auto db = tensor_matmul(a_T, self->grad);
    auto db_red =
        tensor_sum_to_shape(db, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, db_red);
    var_destroy(a_T);
    var_destroy(db_red);
    var_destroy(db);
  }
}

Tensor var_matmul(Tensor a, Tensor b) {

  Tensor tmp = tensor_matmul(a, b);
  if (!tmp)
    return nullptr;

  if ((a->is_tensor_type || !a->requires_grad) &&
      (b->is_tensor_type || !b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, a, b, (VarOp){matmul_backward_fn, nullptr}, nullptr);

  return res;
}
