#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

struct layer_norm_ctx {
  Tensor variance;
  u32 axis;
};

void destroy_layer_norm_ctx(struct layer_norm_ctx *ctx) {
  tensor_destroy(ctx->variance);
}

void layer_norm_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent = self->parent[0];

  struct layer_norm_ctx *ctx = self->ctx;

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    auto tmp = tensor_layer_norm_axis_backward(self->grad, &self->base,
                                               ctx->axis, ctx->variance);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent->base.ndims, parent->base.shape);
    tensor_add_inplace(parent->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

Tensor var_layer_norm_axis(Tensor input, u32 axis) {

  Tensor variance = nullptr;

  Tensor tmp = tensor_layer_norm_axis(input, axis, &variance);
  if (!tmp)
    return nullptr;

  if (input->is_tensor_type || !input->requires_grad)
    return tmp;

  struct layer_norm_ctx *ctx = tmalloc(sizeof(*ctx));
  *ctx = (struct layer_norm_ctx){variance, axis};

  Tensor res = track(tmp);
  var_track_parent(
      res, input, nullptr,
      (VarOp){layer_norm_backward_fn, (void (*)(mem))destroy_layer_norm_ctx},
      ctx);

  return res;
}
