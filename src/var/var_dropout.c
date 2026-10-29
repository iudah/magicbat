#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <stdatomic.h>
#include <stdint.h>

struct dropout_ctx {
  u8 *mask;
  f32 rate;
};

void dropout_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var                 parent = self->parent[0];
  struct dropout_ctx *ctx    = self->ctx;

  if (parent && !parent->base.is_tensor_type && parent->base.requires_grad) {
    if (!parent->grad) {
      parent->grad = tensor_zero(parent->base.ndims, parent->base.shape);
    }
    auto tmp = tensor_dropout_backward(ctx->mask, ctx->rate, self->grad);
    auto tmp_red =
        tensor_sum_to_shape(tmp, parent->base.ndims, parent->base.shape);
    tensor_add_inplace(parent->grad, tmp_red);
    var_destroy(tmp_red);
    var_destroy(tmp);
  }
}

static void dropout_ctx_destroy(mem ctx_ptr) {
  struct dropout_ctx *ctx = ctx_ptr;
  tfree(ctx->mask);
  tfree(ctx);
}

Tensor var_dropout(Tensor var, f32 rate) {
  if (rate <= EPS) {
    atomic_fetch_add(&var->refcount, 1);
    return var;
  }

  rate = rate >= 1 ? 1 - EPS : rate;
  u8    *mask;
  Tensor tmp = tensor_dropout(var, &mask, rate);
  if (!tmp)
    return nullptr;

  if (var->is_tensor_type || !var->requires_grad)
    return tmp;

  struct dropout_ctx *ctx = tcalloc(1, sizeof(*ctx));
  ctx->mask               = mask;
  ctx->rate               = rate;

  Tensor res = track(tmp);
  var_track_parent(res, var, nullptr,
                   (VarOp){dropout_backward_fn, dropout_ctx_destroy}, ctx);

  return res;
}
