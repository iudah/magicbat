#include "tensor.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void bmatmul_backward_fn(Var self) {
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
    auto grad_wrt_a = tensor_bmm_wrt_a(self->grad, &parent_b->base);
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
    // auto a_T = tensor_transpose((Tensor)parent_a);
    auto grad_wrt_b = tensor_bmm_wrt_b(&parent_a->base, self->grad);
    // tensor_matmul(a_T, self->grad);
    auto db_red = grad_wrt_b;
    // tensor_sum_to_shape(db, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, db_red);
    // var_destroy(a_T);
    // var_destroy(db_red);
    var_destroy(grad_wrt_b);
  }
}

Tensor var_bmm(Tensor var_a, Tensor var_b) {

  Tensor tmp = tensor_bmm(var_a, var_b);
  if (!tmp)
    return nullptr;

  if ((var_a->is_tensor_type || !var_a->requires_grad) &&
      (var_b->is_tensor_type || !var_b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, var_a, var_b, (VarOp){bmatmul_backward_fn, nullptr},
                   nullptr);

  return res;
}

void bmatmul_transpose_b_backward_fn(Var self) {
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
    auto grad_wrt_a = tensor_bmm_transpose_b_wrt_a(self->grad, &parent_b->base);
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
    // auto a_T = tensor_transpose((Tensor)parent_a);
    auto grad_wrt_b = tensor_bmm_transpose_b_wrt_b(&parent_a->base, self->grad);
    // tensor_matmul(a_T, self->grad);
    auto db_red = grad_wrt_b;
    // tensor_sum_to_shape(db, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, db_red);
    // var_destroy(a_T);
    // var_destroy(db_red);
    var_destroy(grad_wrt_b);
  }
}

Tensor var_bmm_transpose_b(Tensor var_a, Tensor var_b) {

  Tensor tmp = tensor_bmm_transpose_b(var_a, var_b);
  if (!tmp)
    return nullptr;

  if ((var_a->is_tensor_type || !var_a->requires_grad) &&
      (var_b->is_tensor_type || !var_b->requires_grad))
    return tmp;

  Tensor res = track_replace_untracked(tmp);
  var_track_parent(res, var_a, var_b,
                   (VarOp){bmatmul_transpose_b_backward_fn, nullptr}, nullptr);

  return res;
}
