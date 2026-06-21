#include "fused_attention.h"
#include "fused_attention_backward.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

struct mha_ctx {
  Tensor qkv_contiguous;
  Tensor input;
  f32 *sums;
  f32 *maxs;
  f32 sqrt_d;
};

static void destroy_ctx(struct mha_ctx *ctx) {
  tensor_destroy(ctx->qkv_contiguous);
  tensor_destroy(ctx->input);
  tfree(ctx->sums);
  tfree(ctx->maxs);
}

Tensor tensor_fused_attention(const u32 *shape, const f32 *q_mat,
                              const f32 *k_mat, const f32 *v_mat, f32 sqrt_d,
                              f32 *maxs, f32 *sums);

void multihead_attention_backward_fn(Var self);

static inline void dim_swap(u32 nswaps, u32 *swaps, u32 *shape, u32 *stride) {
  u32 shape_cache[] = {shape[0], shape[1], shape[2], shape[3], shape[4]};
  u32 stride_cache[] = {stride[0], stride[1], stride[2], stride[3], stride[4]};

  switch (nswaps) {
  case 5:
    shape[4] = shape_cache[swaps[4]];
    stride[4] = stride_cache[swaps[4]];
  case 4:
    shape[3] = shape_cache[swaps[3]];
    stride[3] = stride_cache[swaps[3]];
  case 3:
    shape[2] = shape_cache[swaps[2]];
    stride[2] = stride_cache[swaps[2]];
  case 2:
    shape[1] = shape_cache[swaps[1]];
    stride[1] = stride_cache[swaps[1]];
  case 1:
    shape[0] = shape_cache[swaps[0]];
    stride[0] = stride_cache[swaps[0]];
  }
}

Tensor var_multihead_attention(Tensor input, Tensor W_qkv, u32 n_head,
                               f32 mha_d) {

  u32 batch = input->shape[0];
  u32 seq = input->shape[1];
  u32 tokens = input->shape[2];

  // shape(input)=[batch, seq, tokens]
  // shape(W_qkv)=[toxens,3*heads*states]
  Tensor flattened;
  if (!input->is_contiguous) {
    flattened = tensor_reshape(input, 2, (u32[MAX_DIMS]){batch * seq, tokens});
  } else {
    flattened = tensor_view(input);
    flattened->shape[0] = batch * seq;
    flattened->shape[1] = tokens;

    flattened->stride[0] = tokens;
    flattened->stride[1] = 1;

    flattened->ndims = 2;

    flattened->is_contiguous = true;
  }
  // shape(qkv)=[batch, seq,3*heads*states]
  Tensor qkv = tensor_matmul(flattened, W_qkv);
  u32 head_dim = W_qkv->shape[1] / n_head / 3;
#define QKV_DIM 5
  // shape(qkv_reshaped)=[batch, seq, 3, heads, states]
  Tensor qkv_reshaped = qkv;
  qkv_reshaped->ndims = QKV_DIM;
  qkv_reshaped->shape[0] = batch;
  qkv_reshaped->shape[1] = seq;
  qkv_reshaped->shape[2] = 3;
  qkv_reshaped->shape[3] = n_head;
  qkv_reshaped->shape[4] = head_dim;
  qkv_reshaped->stride[0] =
      seq * (qkv_reshaped->stride[1] =
                 3 * (qkv_reshaped->stride[2] =
                          n_head * (qkv_reshaped->stride[3] = head_dim)));
  qkv_reshaped->stride[4] = 1;
  dim_swap(5, (u32[]){2, 0, 3, 1, 4}, qkv_reshaped->shape,
           qkv_reshaped->stride);
  qkv_reshaped->is_contiguous = false;
  // shape(qkv_t)=[3, seq, batch, heads, states]
  // shape(qkv_t)=[3, batch, seq, heads, states]
  // shape(qkv_t)=[3, batch, heads, seq, states]
  Tensor qkv_t = qkv_reshaped;

  Tensor qkv_contiguous = qkv_t;
  tensor_to_contiguous_inplace(qkv_t);

  auto data_len = qkv_contiguous->stride[0];
  f32 *q_mat_data = qkv_contiguous->data->data;
  f32 *k_mat_data = qkv_contiguous->data->data + data_len;
  f32 *v_mat_data = qkv_contiguous->data->data + (data_len * 2);

  u32 cache_len = batch * n_head * seq;
  f32 *maxs = tmalloc(cache_len * sizeof(f32));
  f32 *sums = tcalloc(cache_len, sizeof(f32));

  for (u32 i = 0; i < cache_len; ++i) {
    maxs[i] = -INFINITY;
  }

  auto sqrt_d = sqrtf(mha_d);

  // shape(qkv_t)=[batch * heads * seq, states]
  auto fused_att =
      tensor_fused_attention((u32[4]){batch, n_head, seq, head_dim}, q_mat_data,
                             k_mat_data, v_mat_data, sqrt_d, maxs, sums);

  dim_swap(4, (u32[]){0, 2, 1, 3}, fused_att->shape, fused_att->stride);
  fused_att->is_contiguous = false;
  tensor_to_contiguous_inplace(fused_att);
  fused_att->ndims = 3;
  fused_att->shape[2] *= fused_att->shape[3];
  fused_att->stride[2] = 1;

  if (!fused_att || ((input->is_tensor_type || !input->requires_grad) &&
                     (W_qkv->is_tensor_type || !W_qkv->requires_grad))) {
    tfree(sums);
    tfree(maxs);
    tensor_destroy(flattened);
    return fused_att;
  }

  struct mha_ctx *ctx = tmalloc(sizeof(*ctx));
  *ctx = (struct mha_ctx){qkv_contiguous, .maxs = maxs, .sums = sums,
                          .sqrt_d = sqrt_d, .input = flattened};

  Tensor res = track(fused_att);
  var_track_parent(
      res, input, W_qkv,
      (VarOp){multihead_attention_backward_fn, (void (*)(mem))destroy_ctx},
      ctx);

  return res;
}

Tensor tensor_fused_attention(const u32 *shape, const f32 *q_mat,
                              const f32 *k_mat, const f32 *v_mat, f32 sqrt_d,
                              f32 *maxs, f32 *sums) {
  Tensor fused_att = tensor_new(4, shape);
  auto batch = shape[0] * shape[1];
  auto row = shape[2];
  auto col = shape[2];
  auto com = shape[3];
  auto batch_stride = row * com;

  if (row < BLOCK_SIZE || col < BLOCK_SIZE || com < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_lt_block_size(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &fused_att->data->data[batch_index * batch_stride], sqrt_d,
          &maxs[batch_index * row], &sums[batch_index * row]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_gt_block_size_unbatched(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &fused_att->data->data[batch_index * batch_stride], sqrt_d,
          &maxs[batch_index * row], &sums[batch_index * row]);
    }

  } else {
    fused_attention_gt_block_size(batch, row, col, com, q_mat, k_mat, v_mat,
                                  fused_att->data->data, sqrt_d, maxs, sums);
  }

  return fused_att;
}

static inline void compute_row_sum(const u32 *restrict shape,
                                   const f32 *restrict out,
                                   const f32 *restrict dout,
                                   f32 *restrict res) {
  for (u32 i = 0; i < shape[0]; ++i) {
    for (u32 j = 0; j < shape[1]; ++j) {
      auto i_offset = (i * shape[1]) + j;
      res[i] += out[i_offset] * dout[i_offset];
    }
  }
}

void tensor_fused_attention_backward(const u32 *shape, const f32 *q_mat,
                                     const f32 *k_mat, const f32 *v_mat,
                                     const f32 *r_grad_mat, f32 *q_grad_mat,
                                     f32 *k_grad_mat, f32 *v_grad_mat,
                                     f32 sqrt_d, const f32 *maxs,
                                     const f32 *sums, const f32 *row_sum);
void multihead_attention_backward_fn(Var self) {
  if (self->base.is_tensor_type || !self->base.requires_grad)
    return;

  Var parent_a = self->parent[0];
  Var parent_b = self->parent[1];

  if (!(parent_a && !parent_a->base.is_tensor_type &&
        parent_a->base.requires_grad) &&
      !(parent_b && !parent_b->base.is_tensor_type &&
        parent_b->base.requires_grad))
    return;

  struct mha_ctx *ctx = self->ctx;

  Tensor qkv_contiguous = ctx->qkv_contiguous;

  u32 batch = qkv_contiguous->shape[1];
  u32 n_head = qkv_contiguous->shape[2];
  u32 seq = qkv_contiguous->shape[3];
  u32 head_dim = qkv_contiguous->shape[4];

  u32 full_batch = batch * n_head * seq;

  auto grad = (self->grad);
  auto out = (&self->base);
  out->ndims = grad->ndims = 4;
  out->shape[0] = grad->shape[0] = batch;
  out->shape[1] = grad->shape[1] = seq;
  out->shape[2] = grad->shape[2] = n_head;
  out->shape[3] = grad->shape[3] = head_dim;

  out->stride[0] = grad->stride[0] =
      seq * (out->stride[1] = grad->stride[1] =
                 n_head * (out->stride[2] = grad->stride[2] = head_dim));
  out->stride[3] = grad->stride[3] = 1;

  dim_swap(4, (u32[]){0, 2, 1, 3}, grad->shape, grad->stride);
  dim_swap(4, (u32[]){0, 2, 1, 3}, out->shape, out->stride);

  grad->is_contiguous = false;
  out->is_contiguous = false;

  tensor_to_contiguous_inplace(grad);
  tensor_to_contiguous_inplace(out);

  f32 *row_sum_do_o = tcalloc(full_batch, sizeof(f32));
  compute_row_sum((u32[]){full_batch, head_dim}, self->base.data->data,
                  self->grad->data->data, row_sum_do_o);

  auto data_len = qkv_contiguous->data->nelements / 3;
  f32 *q_mat_data = qkv_contiguous->data->data;
  f32 *k_mat_data = qkv_contiguous->data->data + data_len;
  f32 *v_mat_data = qkv_contiguous->data->data + (data_len * 2);

  f32 *maxs = ctx->maxs;
  f32 *sums = ctx->sums;

  // guarantees zeros
  auto qkv_t_grad = tensor_zero(qkv_contiguous->ndims, qkv_contiguous->shape);
  f32 *q_grad_mat_data = qkv_t_grad->data->data;
  f32 *k_grad_mat_data = qkv_t_grad->data->data + data_len;
  f32 *v_grad_mat_data = qkv_t_grad->data->data + (data_len * 2);

  // shape(qkv_t)=[3, batch, heads, seq, states]
  tensor_fused_attention_backward(
      self->base.shape, q_mat_data, k_mat_data, v_mat_data,
      self->grad->data->data, q_grad_mat_data, k_grad_mat_data, v_grad_mat_data,
      ctx->sqrt_d, maxs, sums, row_sum_do_o);

  // shape(qkv_t)=[3, batch, heads, seq, states]
  // shape(qkv_t)=[batch, 3, heads, seq, states]
  // shape(qkv_t)=[batch, heads, 3, seq, states]
  // shape(qkv_t)=[batch, seq, 3, heads, states]
  dim_swap(5, (u32[]){1, 3, 0, 2, 4}, qkv_t_grad->shape, qkv_t_grad->stride);
  /* Tensor qkv_grad = tensor_transpose_dims(
       qkv_t_grad, 3,
       (TensorDimSwap[]){
           {.src = 0, .dest = 1}, {.src = 1, .dest = 2}, {.src = 1, .dest =
     3}});
 */
  Tensor qkv_grad = qkv_t_grad;
  tensor_to_contiguous_inplace(qkv_grad);

  qkv_grad->shape[0] = batch * seq;
  qkv_grad->shape[1] = 3 * n_head * head_dim;
  qkv_grad->stride[0] = 3 * n_head * head_dim;
  qkv_grad->stride[1] = 1;
  qkv_grad->ndims = 2;
  qkv_grad->is_contiguous = true;

  if (parent_a && !parent_a->base.is_tensor_type &&
      parent_a->base.requires_grad) {
    if (!parent_a->grad) {
      parent_a->grad = tensor_zero(parent_a->base.ndims, parent_a->base.shape);
    }
    // auto b_T = tensor_transpose((Tensor)parent_b);
    auto grad_wrt_a = tensor_matmul_wrt_a(qkv_grad, &parent_b->base);

    auto tokens = parent_b->base.shape[0];

    grad_wrt_a->ndims = 3;
    grad_wrt_a->shape[0] = batch;
    grad_wrt_a->shape[1] = seq;
    grad_wrt_a->shape[2] = tokens;

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
    auto grad_wrt_b = tensor_matmul_wrt_b(ctx->input, qkv_grad);
    // tensor_multihead_attention(a_T, self->grad);
    auto db_red = grad_wrt_b;
    // tensor_sum_to_shape(db, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, db_red);
    // var_destroy(a_T);
    // var_destroy(db_red);
    var_destroy(grad_wrt_b);
  }

  tensor_destroy(qkv_grad);
  tfree(row_sum_do_o);
}

void tensor_fused_attention_backward(const u32 *shape, const f32 *q_mat,
                                     const f32 *k_mat, const f32 *v_mat,
                                     const f32 *r_grad_mat, f32 *q_grad_mat,
                                     f32 *k_grad_mat, f32 *v_grad_mat,
                                     f32 sqrt_d, const f32 *maxs,
                                     const f32 *sums, const f32 *row_sum) {

  auto batch = shape[0] * shape[1];
  auto row = shape[2];
  auto col = shape[2];
  auto com = shape[3];
  auto batch_stride = row * com;

  if (row <= BLOCK_SIZE || col <= BLOCK_SIZE || com <= BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_backward_lt_block_size(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &q_grad_mat[batch_index * batch_stride],
          &k_grad_mat[batch_index * batch_stride],
          &v_grad_mat[batch_index * batch_stride],
          &r_grad_mat[batch_index * batch_stride], sqrt_d,
          &maxs[batch_index * row], &sums[batch_index * row],
          &row_sum[batch_index * row]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_backward_gt_block_size_unbatched(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &q_grad_mat[batch_index * batch_stride],
          &k_grad_mat[batch_index * batch_stride],
          &v_grad_mat[batch_index * batch_stride],
          &r_grad_mat[batch_index * batch_stride], sqrt_d,
          &maxs[batch_index * row], &sums[batch_index * row],
          &row_sum[batch_index * row]);
    }

  } else {
    fused_attention_backward_gt_block_size(
        batch, row, col, com, q_mat, k_mat, v_mat, q_grad_mat, k_grad_mat,
        v_grad_mat, r_grad_mat, sqrt_d, maxs, sums, row_sum);
  }
}
