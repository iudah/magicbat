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

    flattened->shape[1] = tokens;
    flattened->shape[0] = 1;

    flattened->ndims = 2;

    flattened->is_contiguous = false;
  }
  // shape(qkv)=[batch, seq,3*heads*states]
  Tensor qkv = tensor_matmul(flattened, W_qkv);
  u32 head_dim = W_qkv->shape[1] / n_head;
#define QKV_DIM 5
  // shape(qkv_reshaped)=[batch, seq, 3, heads, states]
  Tensor qkv_reshaped = qkv;
  qkv_reshaped->ndims = QKV_DIM;
  qkv_reshaped->shape[0] = batch;
  qkv_reshaped->shape[1] = seq;
  qkv_reshaped->shape[2] = 3;
  qkv_reshaped->shape[3] = n_head;
  qkv_reshaped->shape[4] = head_dim;
  // shape(qkv_t)=[3, seq, batch, heads, states]
  // shape(qkv_t)=[3, batch, seq, heads, states]
  // shape(qkv_t)=[3, batch, heads, seq, states]
  Tensor qkv_t = tensor_transpose_dims(
      qkv_reshaped, 3,
      (TensorDimSwap[]){
          {.src = 2, .dest = 0}, {.src = 1, .dest = 2}, {.src = 2, .dest = 3}});

  Tensor qkv_contiguous = qkv_t;
  tensor_to_contiguous_inplace(qkv_t);

  auto data_len = qkv_contiguous->data->nelements / 3;
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

  tensor_destroy(qkv_reshaped);
  tensor_destroy(qkv);

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
  auto batch_stride = row * col;

  if (row < BLOCK_SIZE || col < BLOCK_SIZE || com < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_lt_block_size(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &fused_att->data->data[batch_index * fused_att->stride[0]], sqrt_d,
          &maxs[batch_index * row], &sums[batch_index * row]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      fused_attention_gt_block_size_unbatched(
          row, col, com, &q_mat[batch_index * batch_stride],
          &k_mat[batch_index * batch_stride],
          &v_mat[batch_index * batch_stride],
          &fused_att->data->data[batch_index * fused_att->stride[0]], sqrt_d,
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
        parent_a->base.requires_grad) ||
      !(parent_b && !parent_b->base.is_tensor_type &&
        parent_b->base.requires_grad))
    return;

  u32 batch = self->base.shape[0];
  u32 n_head = self->base.shape[1];
  u32 seq = self->base.shape[2];
  u32 head_dim = self->base.shape[3];

  u32 full_batch = batch * n_head * seq;

  f32 *row_sum_do_o = tcalloc(full_batch, sizeof(f32));
  compute_row_sum((u32[]){full_batch, head_dim}, self->base.data->data,
                  self->grad->data->data, row_sum_do_o);

  struct mha_ctx *ctx = self->ctx;

  Tensor qkv_contiguous = ctx->qkv_contiguous;

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
  Tensor qkv_grad = tensor_transpose_dims(
      qkv_t_grad, 3,
      (TensorDimSwap[]){
          {.src = 0, .dest = 1}, {.src = 1, .dest = 2}, {.src = 1, .dest = 3}});

  tensor_to_contiguous_inplace(qkv_grad);

  qkv_grad->shape[0] = batch * seq * 3 * n_head;
  qkv_grad->shape[1] = head_dim;
  qkv_grad->stride[0] = head_dim;
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

  tensor_destroy(qkv_t_grad);
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
  auto batch_stride = row * col;

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
