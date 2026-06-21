#include "tensor.h"
#include "tensor_memory.h"
#include "var.h"
#include "var_prot.h"

void ms_backward_fn(Var self) {
  Var var_a = (Var)self->parent[0];
  Var var_b = (Var)self->parent[1];

  if (self->base.is_tensor_type || !self->base.requires_grad ||
      (!(var_a && !var_a->base.is_tensor_type && var_a->base.requires_grad) &&
       !(var_b && !var_b->base.is_tensor_type && var_b->base.requires_grad)))
    return;

  auto dms = self->grad ? self->grad->data->data[0] : 1;
  auto dsub = tensor_mean_sqr_all_backward(dms, (*(Tensor *)self->ctx));

  if (var_a && !var_a->base.is_tensor_type && var_a->base.requires_grad) {
    if (!var_a->grad) {
      var_a->grad = tensor_zero(var_a->base.ndims, var_a->base.shape);
    }
    auto tmp = tensor_sum_to_shape(dsub, var_a->base.ndims, var_a->base.shape);
    tensor_add_inplace(var_a->grad, tmp);
    var_destroy(tmp);
  }

  tensor_negate_inplace(dsub);
  if (var_b && !var_b->base.is_tensor_type && var_b->base.requires_grad) {
    if (!var_b->grad) {
      var_b->grad = tensor_zero(var_b->base.ndims, var_b->base.shape);
    }
    auto tmp = tensor_sum_to_shape(dsub, var_b->base.ndims, var_b->base.shape);
    tensor_add_inplace(var_b->grad, tmp);
    if (tmp != dsub)
      tensor_destroy(dsub);
    var_destroy(tmp);
  }
}

void end_ctx(Tensor *tensor) { tensor_destroy(*tensor); }

Tensor var_mse_loss(Tensor pred, Tensor target) {
  auto var = tensor_sub(pred, target);
  if (!var)
    return nullptr;

  auto tmp = tensor_new(2, (u32[]){1, 1});
  tmp->data->data[0] = tensor_mean_sqr_all(var);
  if (!tmp)
    return nullptr;

  if ((pred->is_tensor_type || !pred->requires_grad)

      && (target->is_tensor_type || !target->requires_grad)) {
    tensor_destroy(var);
    return tmp;
  }

  Tensor *ctx = tmalloc(sizeof(*ctx));
  *ctx = var;

  Tensor res = track(tmp);
  var_track_parent(res, pred, target,
                   (VarOp){ms_backward_fn, (void (*)(mem))end_ctx}, ctx);

  return res;
}
