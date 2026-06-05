#include "tensor.h"
#include "tensor_prot.h"

static inline bool matmul_tensors_invalid(const Tensor tensor_a,
                                          const Tensor tensor_b) {
  TASSERT(tensor_a && tensor_b && tensor_a->ndims == 2 &&
          tensor_b->ndims == 2 && "Tensors dimensions not 2");
  TASSERT(tensor_a && tensor_b && tensor_a->shape[1] == tensor_b.shape[0] &&
          "Tensors common size are mismatched.");
  return !tensor_a || !tensor_b || tensor_b->ndims != 2 ||
         tensor_a->ndims != 2 || tensor_a->shape[1] != tensor_b->shape[0];
}

Tensor tensor_matmul(const Tensor tensor_a, const Tensor tensor_b) {
  if (matmul_tensors_invalid(tensor_a, tensor_b))
    return nullptr;

  u32 row = tensor_a->shape[0];
  u32 col = tensor_b->shape[1];
  u32 com = tensor_a->shape[1];

  // guarantees zeros
  Tensor product = tensor_new(2, (u32[]){row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_a->is_contiguous || !tensor_b->is_contiguous)
    goto non_contiguous_matmul;

  for (u32 i = 0; i < row; ++i) {
    float *a_data = &tensor_a->data->data[i * com];
    float *p_data = &product->data->data[i * col];
    for (u32 j = 0; j < com; ++j) {
      float *b_data = &tensor_b->data->data[j * col];
      float a_val = a_data[j];
      for (u32 k = 0; k < col; ++k) {
        p_data[k] += a_val * b_data[k];
      }
    }
  }

  goto return_statement;

non_contiguous_matmul:
  for (u32 i = 0; i < row; ++i) {
    float *p_data = &product->data->data[i * col];
    for (u32 j = 0; j < com; ++j) {
      for (u32 k = 0; k < col; ++k) {
        auto a_val = tensor_get(tensor_a, (u32[MAX_DIMS]){i, j});
        auto b_val = tensor_get(tensor_b, (u32[MAX_DIMS]){j, k});
        p_data[k] += a_val * b_val;
      }
    }
  }

return_statement:
  return product;
}

Tensor tensor_matmul_wrt_a(const Tensor tensor_grad, const Tensor tensor_b) {
  if (!tensor_grad || !tensor_b)
    return nullptr;
  if (tensor_b->ndims != 2 || tensor_grad->ndims != 2 ||
      tensor_grad->shape[1] != tensor_b->shape[1])
    return nullptr;

  u32 row = tensor_grad->shape[0];
  u32 col = tensor_b->shape[0];
  u32 com = tensor_grad->shape[1];

  // guarantees zeros
  Tensor product = tensor_new(2, (u32[]){row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_grad->is_contiguous || !tensor_b->is_contiguous)
    goto non_contiguous_matmul;

  for (u32 i = 0; i < row; ++i) {
    f32 *a_data = &tensor_grad->data->data[i * com];
    f32 *p_data = &product->data->data[i * col];
    for (u32 k = 0; k < col; ++k) {
      f32 *b_data = &tensor_b->data->data[k * com];
      f32 sum = 0;
      for (u32 j = 0; j < com; ++j) {
        f32 a_val = a_data[j];
        f32 b_val = b_data[j];
        sum += a_val * b_val;
      }
      p_data[k] = sum;
    }
  }

  goto return_statement;

non_contiguous_matmul:
  for (u32 i = 0; i < row; ++i) {
    f32 *p_data = &product->data->data[i * col];
    for (u32 k = 0; k < col; ++k) {
      f32 sum = 0;
      for (u32 j = 0; j < com; ++j) {
        f32 a_val = tensor_get(tensor_grad, (u32[MAX_DIMS]){i, j});
        f32 b_val = tensor_get(tensor_b, (u32[MAX_DIMS]){k, j});
        sum += a_val * b_val;
      }
      p_data[k] = sum;
    }
  }

return_statement:
  return product;
}

Tensor tensor_matmul_wrt_b(const Tensor tensor_a, const Tensor tensor_grad) {
  if (!tensor_a || !tensor_grad)
    return nullptr;
  if (tensor_grad->ndims != 2 || tensor_a->ndims != 2 ||
      tensor_a->shape[0] != tensor_grad->shape[0])
    return nullptr;

  u32 row = tensor_a->shape[1];
  u32 col = tensor_grad->shape[1];
  u32 com = tensor_a->shape[0];

  // guarantees zeros
  Tensor product = tensor_new(2, (u32[]){row, col});
  if (product == nullptr)
    return nullptr;

  if (!tensor_a->is_contiguous || !tensor_grad->is_contiguous)
    goto non_contiguous_matmul;

  for (u32 j = 0; j < com; ++j) {
    float *a_data = &tensor_a->data->data[j * row];
    float *b_data = &tensor_grad->data->data[j * col];
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[i * col];
      for (u32 k = 0; k < col; ++k) {
        p_data[k] += a_data[i] * b_data[k];
      }
    }
  }

  goto return_statement;

non_contiguous_matmul:
  for (u32 j = 0; j < com; ++j) {
    for (u32 i = 0; i < row; ++i) {
      float *p_data = &product->data->data[i * col];
      for (u32 k = 0; k < col; ++k) {

        f32 a_val = tensor_get(tensor_a, (u32[MAX_DIMS]){j, i});
        f32 b_val = tensor_get(tensor_grad, (u32[MAX_DIMS]){j, k});

        p_data[k] += a_val * b_val;
      }
    }
  }

return_statement:
  return product;
}
