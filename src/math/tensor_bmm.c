#include "matmul.h"
#include "matmul_batched_tile.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "tensor_prot.h"
#include <stdint.h>

static inline bool bmm_tensors_invalid(const Tensor tensor_a,
                                       const Tensor tensor_b) {
  TASSERT(tensor_a && tensor_b && tensor_a->ndims == 3 &&
          tensor_b->ndims == 3 && "Tensors dimensions not 3");
  TASSERT(tensor_a && tensor_b && tensor_a->shape[0] == tensor_b->shape[0] &&
          tensor_a->shape[2] == tensor_b->shape[1] &&
          "Tensors common size are mismatched.");
  return !tensor_a || !tensor_b || tensor_b->ndims != 3 ||
         tensor_a->ndims != 3 || tensor_a->shape[0] != tensor_b->shape[0] ||
         tensor_a->shape[2] != tensor_b->shape[1];
}

void bmm_non_contiguous(Tensor tensor_a, Tensor tensor_b, Tensor product,
                        u32 batch, u32 row, u32 col, u32 com) {
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 j = 0; j < com; ++j) {
        for (u32 k = 0; k < col; ++k) {
          auto a_val = tensor_get(tensor_a, (u32[MAX_DIMS]){batch_index, i, j});
          auto b_val = tensor_get(tensor_b, (u32[MAX_DIMS]){batch_index, j, k});
          p_data[k] += a_val * b_val;
        }
      }
    }
  }
}

Tensor tensor_bmm(const Tensor tensor_a, const Tensor tensor_b) {
  if (bmm_tensors_invalid(tensor_a, tensor_b))
    return nullptr;

  u32 batch = tensor_a->shape[0];
  u32 row = tensor_a->shape[1];
  u32 col = tensor_b->shape[2];
  u32 com = tensor_a->shape[2];

  // guarantees zeros
  Tensor product = tensor_new(3, (u32[]){batch, row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_a->is_contiguous || !tensor_b->is_contiguous)
    goto non_contiguous_matmul;

  if (row < BLOCK_SIZE || col < BLOCK_SIZE || com < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_lt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_gt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }

  } else {
    bmatmul_gt_block_size(batch, row, col, com, tensor_a->data->data,
                          tensor_b->data->data, product->data->data);
  }

  goto return_statement;

non_contiguous_matmul:
  bmm_non_contiguous(tensor_a, tensor_b, product, batch, row, col, com);

return_statement:
  return product;
}

Tensor tensor_bmm_wrt_a(const Tensor tensor_grad, const Tensor tensor_b) {
  if (bmm_tensors_invalid(tensor_grad, tensor_b))
    return nullptr;

  u32 batch = tensor_grad->shape[0];
  u32 row = tensor_grad->shape[1];
  u32 col = tensor_b->shape[1];
  u32 com = tensor_grad->shape[2];

  // guarantees zeros
  Tensor product = tensor_new(2, (u32[]){row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_grad->is_contiguous || !tensor_b->is_contiguous)
    goto non_contiguous_matmul;

  if (row < BLOCK_SIZE || col < BLOCK_SIZE || com < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_transpose_b_lt_block_size(
          row, col, com,
          &tensor_grad->data->data[batch_index * tensor_grad->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_transpose_b_gt_block_size(
          row, col, com,
          &tensor_grad->data->data[batch_index * tensor_grad->stride[0]],
          &tensor_b->data->data[batch_index * tensor_b->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }

  } else {
    bmatmul_transpose_b_gt_block_size(
        batch, row, col, com, tensor_grad->data->data, tensor_b->data->data,
        product->data->data);
  }

  goto return_statement;

non_contiguous_matmul:
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 i = 0; i < row; ++i) {
      f32 *p_data = &product->data->data[(batch_index * row + i) * col];
      for (u32 k = 0; k < col; ++k) {
        f32 sum = 0;
        for (u32 j = 0; j < com; ++j) {
          f32 a_val =
              tensor_get(tensor_grad, (u32[MAX_DIMS]){batch_index, i, j});
          f32 b_val = tensor_get(tensor_b, (u32[MAX_DIMS]){batch_index, k, j});
          sum += a_val * b_val;
        }
        p_data[k] = sum;
      }
    }
  }

return_statement:
  return product;
}

Tensor tensor_bmm_wrt_b(const Tensor tensor_a, const Tensor tensor_grad) {
  if (bmm_tensors_invalid(tensor_a, tensor_grad))
    return nullptr;

  u32 batch = tensor_a->shape[0];
  u32 row = tensor_a->shape[2];
  u32 col = tensor_grad->shape[2];
  u32 com = tensor_a->shape[1];

  // guarantees zeros
  Tensor product = tensor_new(3, (u32[]){batch, row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_a->is_contiguous || !tensor_grad->is_contiguous)
    goto non_contiguous_matmul;

  if (row < BLOCK_SIZE || col < BLOCK_SIZE || com < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_transpose_a_lt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_grad->data->data[batch_index * tensor_grad->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }
  } else if (batch < BLOCK_SIZE) {

    for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
      matmul_transpose_a_gt_block_size(
          row, col, com,
          &tensor_a->data->data[batch_index * tensor_a->stride[0]],
          &tensor_grad->data->data[batch_index * tensor_grad->stride[0]],
          &product->data->data[batch_index * product->stride[0]]);
    }

  } else {
    bmatmul_transpose_a_gt_block_size(
        batch, row, col, com, tensor_a->data->data, tensor_grad->data->data,
        product->data->data);
  }

  goto return_statement;

non_contiguous_matmul:
  for (u32 batch_index = 0; batch_index < batch; ++batch_index) {
    for (u32 j = 0; j < com; ++j) {
      for (u32 i = 0; i < row; ++i) {
        float *p_data = &product->data->data[(batch_index * row + i) * col];
        for (u32 k = 0; k < col; ++k) {

          f32 a_val = tensor_get(tensor_a, (u32[MAX_DIMS]){batch_index, j, i});
          f32 b_val =
              tensor_get(tensor_grad, (u32[MAX_DIMS]){batch_index, j, k});

          p_data[k] += a_val * b_val;
        }
      }
    }
  }

return_statement:
  return product;
}
