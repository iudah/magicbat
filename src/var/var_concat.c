#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <string.h>

struct concat_ctx {
  struct bound {
    u32 start;
    u32 stop;
  } bound[2];
  u32 axis;
};

void slice_concat_grad(Var var, Tensor grad, u32 axis, struct bound bound) {
  TensorSlice slice[MAX_DIMS];
  for (u32 i = 0; i < var->base.ndims; ++i) {
    slice[i] = (TensorSlice){.start = 0, .stop = var->base.shape[i], 1};
  }
  slice[axis] = (TensorSlice){.start = bound.start, .stop = bound.stop};

  auto tmp = tensor_slice(grad, slice);

  // Var does not have data storage, then var should be concat shell.
  if (!var->base.data) {
    var->grad = tmp;
  } else {
    if (!var->grad) {
      var->grad = tensor_zero(var->base.ndims, var->base.shape);
    }
    tensor_add_inplace(var->grad, tmp);
    var_destroy(tmp);
  }
}

void concat_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var var_a = (Var)self->parent[0];
  Var var_b = (Var)self->parent[1];

  struct concat_ctx *ctx = self->ctx;

  if (var_a && !var_a->base.is_tensor_type && var_a->base.requires_grad) {
    slice_concat_grad(var_a, self->grad, ctx->axis, ctx->bound[0]);
  }
  if (var_b && !var_b->base.is_tensor_type && var_b->base.requires_grad) {
    slice_concat_grad(var_b, self->grad, ctx->axis, ctx->bound[1]);
  }
}

Var var_concat_tree(Var parent_a, Var parent_b, struct concat_ctx slices) {
  Var concat = tcalloc(1, sizeof(*concat));
  concat->parent[0] = parent_a;
  concat->parent[1] = parent_b;

  struct concat_ctx *ctx = concat->ctx;
  concat->ctx = tmalloc(2 * sizeof(*ctx));

  if (parent_a) {
    ctx->bound[0].start = slices.bound[0].start;
    ctx->bound[0].stop = slices.bound[0].stop;
  }

  if (parent_b) {
    ctx->bound[1].start = slices.bound[1].start;
    ctx->bound[1].stop = slices.bound[1].stop;
  }

  var_track_parent(&concat->base, &parent_a->base, &parent_b->base,
                   (VarOp){.backward = concat_backward_fn,
                           .destroy_ctx = (void (*)(mem))(void *)tfree},
                   ctx);

  return concat;
}

Tensor var_concat(u32 ntensor, Tensor *tensors, u32 axis) {
  Tensor concat = tensor_concat(ntensor, tensors, axis);

  if (ntensor == 1)
    return concat;

  Var joins[ntensor];
  u32 starts[ntensor];
  u32 stops[ntensor];
  memcpy(joins, tensors, ntensor * sizeof(Tensor));
  u32 count = ntensor;
  u32 stop = 0;

  for (u32 i = 0, j = 0; i < ntensor; ++i) {
    auto start = stop;
    stop += joins[i]->base.shape[axis];

    if (joins[i]->base.is_tensor_type || !joins[i]->base.requires_grad) {
      count--;
      continue;
    }

    joins[j] = joins[i];
    starts[j] = start;
    stops[j] = stop;
    ++j;
  }

  while (count > 2) {
    count /= 2;
    for (u32 i = 0; i < count; ++i) {
      joins[i] = var_concat_tree(
          joins[i * 2], joins[(i * 2) + 1],
          (struct concat_ctx){
              .bound = {[0] = {.start = starts[i * 2], .stop = stops[i * 2]},
                        [1] = {.start = starts[(i * 2) + 1],
                               .stop = stops[(i * 2) + 1]}},
              .axis = axis});
      starts[i] = starts[i * 2];
      stops[i] = stops[i * 2];
    }
  }

  struct concat_ctx *ctx = tmalloc(sizeof(*ctx));

  ctx->axis = axis;
  ctx->bound[0] = (typeof(*ctx->bound)){.start = starts[0], .stop = stops[0]};
  if (count == 2)
    ctx->bound[1] = (typeof(*ctx->bound)){.start = starts[1], .stop = stops[1]};

  auto res = track(concat);
  var_track_parent(
      res, (Tensor)(joins[0]), (Tensor)((count == 1) ? nullptr : joins[1]),
      (VarOp){concat_backward_fn, (void (*)(mem))(void *)tfree}, ctx);

  return res;
}
