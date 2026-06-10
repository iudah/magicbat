#include "fused_attention.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "tensor_memory.h"
#include "var_prot.h"
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

void multihead_attention_backward_fn(Var self) {
#if 0
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
    auto grad_wrt_a =
        tensor_multihead_attention_wrt_a(self->grad, &parent_a->base);
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
    auto grad_wrt_b =
        tensor_multihead_attention_wrt_b(&parent_a->base, self->grad);
    // tensor_multihead_attention(a_T, self->grad);
    auto db_red = grad_wrt_b;
    // tensor_sum_to_shape(db, parent_b->base.ndims, parent_b->base.shape);
    tensor_add_inplace(parent_b->grad, db_red);
    // var_destroy(a_T);
    // var_destroy(db_red);
    var_destroy(grad_wrt_b);
  }
#endif
}

Tensor tensor_fused_attention(const u32 *shape, const f32 *q_mat,
                              const f32 *k_mat, const f32 *v_mat, f32 sqrt_d,
                              f32 *maxs, f32 *sums);

Tensor var_multihead_attention(Tensor input, Tensor W_qkv, u32 n_head,
                               f32 mha_d) {
  Tensor flattened = tensor_reshape(
      input, 2,
      (u32[MAX_DIMS]){input->shape[0] * input->shape[1], input->shape[2]});
  Tensor qkv = tensor_matmul(flattened, W_qkv);
  u32 head_dim = W_qkv->shape[1] / n_head;
#define QKV_DIM 5
  Tensor qkv_reshaped = tensor_reshape(
      qkv, QKV_DIM,
      (u32[MAX_DIMS]){input->shape[0], input->shape[1], 3, n_head, head_dim});
  Tensor qkv_t = tensor_transpose_dims(
      qkv_reshaped, 3,
      (TensorDimSwap[]){
          {.src = 2, .dest = 0}, {.src = 1, .dest = 2}, {.src = 1, .dest = 3}});

  Tensor qkv_contiguous = tensor_to_contiguous(qkv_t);

  /*
  Tensor q_mat = tensor_new(
      2, (u32[]){qkv_contiguous->shape[1] * qkv_contiguous->shape[2] * n_head,
                 head_dim});
  Tensor k_mat = tensor_new(
      2, (u32[]){qkv_contiguous->shape[1] * qkv_contiguous->shape[2] * n_head,
                 head_dim});
  Tensor v_mat = tensor_new(
      2, (u32[]){qkv_contiguous->shape[1] * qkv_contiguous->shape[2] * n_head,
                 head_dim});

  f32 *qkv_data[] = {q_mat->data->data, k_mat->data->data, v_mat->data->data};
*/

  auto data_len = qkv_contiguous->data->nelements / 3;
  f32 *q_mat_data = qkv_contiguous->data->data;
  f32 *k_mat_data = qkv_contiguous->data->data + data_len;
  f32 *v_mat_data = qkv_contiguous->data->data + (data_len * 2);

  f32 *maxs = tmalloc(data_len / head_dim * sizeof(f32));
  f32 *sums = tcalloc(data_len / head_dim, sizeof(f32));

  for (u32 i = 0; i < data_len / head_dim; ++i) {
    maxs[i] = -INFINITY;
  }

  auto fused_att =
      tensor_fused_attention((u32[]){data_len / head_dim, head_dim}, q_mat_data,
                             k_mat_data, v_mat_data, sqrtf(mha_d), maxs, sums);

  /*
  if (!tmp)
    return nullptr;

  if ((input->is_tensor_type || !input->requires_grad) &&
      (W_qkv->is_tensor_type || !W_qkv->requires_grad))
    return tmp;

  Tensor res = track(tmp);
  var_track_parent(res, input, W_qkv,
                   (VarOp){multihead_attention_backward_fn, nullptr}, nullptr);

 */
  return fused_att;
}

Tensor tensor_fused_attention(const u32 *shape, const f32 *q_mat,
                              const f32 *k_mat, const f32 *v_mat, f32 sqrt_d,
                              f32 *maxs, f32 *sums) {
  Tensor fused_att = tensor_new(2, shape);
  auto batch = shape[0] / shape[1];
  auto row = shape[1];
  auto col = shape[1];
  auto com = shape[1];
  auto batch_stride = row * col;

#define TILE_SIZE 16

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
