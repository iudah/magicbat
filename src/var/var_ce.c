#include "tensor.h"
#include "tensor_memory.h"
#include "tensor_prot.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

void cross_entropy_oh_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  i32 *axis_ptr = self->ctx;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_ONE_HOT, *axis_ptr);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }
}

Tensor var_cross_entropy_one_hot(Tensor logits, Tensor targets, i32 axis) {

  Tensor tmp = tensor_cross_entropy_from_logits_axis(logits, targets,
                                                     TENSOR_CE_ONE_HOT, axis);

  if (!tmp)
    return nullptr;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  i32 *axis_ptr = tmalloc(sizeof(*axis_ptr));
  *axis_ptr = axis;
  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_oh_backward_fn, nullptr}, axis_ptr);

  return res;
}

void cross_entropy_pr_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  i32 *axis_ptr = self->ctx;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_PROBS, *axis_ptr);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }
}

Tensor var_cross_entropy_probs(Tensor logits, Tensor targets, i32 axis) {

  Tensor tmp = tensor_cross_entropy_from_logits_axis(logits, targets,
                                                     TENSOR_CE_PROBS, axis);

  if (!tmp)
    return nullptr;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  i32 *axis_ptr = tmalloc(sizeof(*axis_ptr));
  *axis_ptr = axis;
  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_pr_backward_fn, nullptr}, axis_ptr);

  return res;
}

void cross_entropy_idx_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  i32 *axis_ptr = self->ctx;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_INDEXED, *axis_ptr);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
#ifdef INDEXED_CROSS_ENTROPY_COMPLETED
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }
#endif
}

Tensor var_cross_entropy_indexed(Tensor logits, Tensor targets, i32 axis) {

  Tensor tmp = tensor_cross_entropy_from_logits_axis(logits, targets,
                                                     TENSOR_CE_INDEXED, axis);

  if (!tmp)
    return nullptr;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  i32 *axis_ptr = tmalloc(sizeof(*axis_ptr));
  *axis_ptr = axis;
  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_idx_backward_fn, nullptr}, axis_ptr);

  return res;
}

struct ce_loss {
  u32 axis;
  u32 mean_divider;
};

void cross_entropy_loss_oh_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  struct ce_loss *ctx = self->ctx;

  auto tmp = *self->grad->data->data;
  self->grad->data->data[0] /= ctx->mean_divider;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_ONE_HOT, ctx->axis);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }

  self->grad->data->data[0] = tmp;
}

Tensor var_cross_entropy_loss_one_hot(Tensor logits, Tensor targets, i32 axis) {

  Tensor cross_ent = tensor_cross_entropy_from_logits_axis(
      logits, targets, TENSOR_CE_ONE_HOT, axis);
  auto loss = tensor_mean_all(cross_ent);

  auto tmp = tensor_new(2, (u32[]){1, 1});
  if (!tmp)
    return nullptr;

  tmp->data->data[0] = loss;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  struct ce_loss *ctx = tmalloc(sizeof(*ctx));
  ctx->axis = axis;
  ctx->mean_divider = cross_ent->data->nelements;

  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_loss_oh_backward_fn, nullptr}, ctx);

  return res;
}

void cross_entropy_loss_pr_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  struct ce_loss *ctx = self->ctx;

  auto tmp = *self->grad->data->data;
  self->grad->data->data[0] /= ctx->mean_divider;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_PROBS, ctx->axis);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }

  *self->grad->data->data = tmp;
}

Tensor var_cross_entropy_loss_probs(Tensor logits, Tensor targets, i32 axis) {

  Tensor cross_ent = tensor_cross_entropy_from_logits_axis(
      logits, targets, TENSOR_CE_PROBS, axis);
  auto loss = tensor_mean_all(cross_ent);

  auto tmp = tensor_new(2, (u32[]){1, 1});
  if (!tmp)
    return nullptr;

  tmp->data->data[0] = loss;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  struct ce_loss *ctx = tmalloc(sizeof(*ctx));
  ctx->axis = axis;
  ctx->mean_divider = cross_ent->data->nelements;

  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_loss_pr_backward_fn, nullptr}, ctx);

  return res;
}

void cross_entropy_loss_idx_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var logits = (Var)self->parent[0];
  Var target = (Var)self->parent[1];

  struct ce_loss *ctx = self->ctx;

  auto tmp = *self->grad->data->data;
  self->grad->data->data[0] /= ctx->mean_divider;

  if (logits && !logits->base.is_tensor_type && logits->base.requires_grad) {
    if (!logits->grad) {
      logits->grad = tensor_zero(logits->base.ndims, logits->base.shape);
    }
    auto tmp = tensor_cross_entropy_from_logits_axis_backward(
        &logits->base, &target->base, self->grad, TENSOR_CE_INDEXED, ctx->axis);
    auto tmp_red =
        tensor_sum_to_shape(tmp, logits->base.ndims, logits->base.shape);
    tensor_add_inplace(logits->grad, tmp_red);
    var_destroy(tmp);
  }
#ifdef INDEXED_CROSS_ENTROPY_COMPLETED
  if (target && !target->base.is_tensor_type && target->base.requires_grad) {
    if (!target->grad) {
      target->grad = tensor_zero(target->base.ndims, target->base.shape);
    }
    auto tmp = tensor_mul(&logits->base, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, target->base.ndims, target->base.shape);
    tensor_add_inplace(target->grad, tmp_red);
    var_destroy(tmp_red);
  }
#endif

  *self->grad->data->data = tmp;
}

Tensor var_cross_entropy_loss_indexed(Tensor logits, Tensor targets, i32 axis) {

  Tensor cross_ent = tensor_cross_entropy_from_logits_axis(
      logits, targets, TENSOR_CE_INDEXED, axis);
  auto loss = tensor_mean_all(cross_ent);

  auto tmp = tensor_new(2, (u32[]){1, 1});
  if (!tmp)
    return nullptr;

  tmp->data->data[0] = loss;

  if ((logits->is_tensor_type || !logits->requires_grad) &&
      (targets->is_tensor_type || !targets->requires_grad))
    return tmp;

  struct ce_loss *ctx = tmalloc(sizeof(*ctx));
  ctx->axis = axis;
  ctx->mean_divider = cross_ent->data->nelements;

  Tensor res = track(tmp);
  var_track_parent(res, logits, targets,
                   (VarOp){cross_entropy_loss_idx_backward_fn, nullptr}, ctx);

  return res;
}
