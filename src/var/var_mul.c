#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void mul_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent_a = self->parent[0];
  Var parent_b = self->parent[1];

  if (parent_a && !parent_a->base.is_tensor_type &&
      parent_a->base.requires_grad) {
    if (!parent_a->grad) {
      parent_a->grad = tensor_zero(parent_a->base.ndims, parent_a->base.shape);
    }
    auto tmp = tensor_mul((Tensor)parent_b, self->grad);
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
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

Tensor var_mul(Tensor var_a, Tensor var_b) {

  Tensor tmp = tensor_mul(var_a, var_b);
  if (!tmp)
    return nullptr;

  if ((var_a->is_tensor_type || !var_a->requires_grad) &&
      (var_b->is_tensor_type || !var_b->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, var_a, var_b, (VarOp){mul_backward_fn, nullptr},
                   nullptr);

  return res;
}

void scale_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent = self->parent[0];

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    auto tmp = tensor_scale(self->grad, *(f32 *)self->ctx);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent->base.ndims, parent->base.shape);
    tensor_add_inplace(parent->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

Tensor var_scale(Tensor var, f32 scalar) {

  Tensor tmp = tensor_scale(var, scalar);
  if (!tmp)
    return nullptr;

  if ((var->is_tensor_type || !var->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  f32 *ctx = tmalloc(sizeof(f32));
  *ctx = scalar;
  var_track_parent(res, var, nullptr, (VarOp){scale_backward_fn, nullptr},
                   (mem)ctx);

  return res;
}
