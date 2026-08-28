#include "tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float mul(float val_a, float val_b, float UNUSED_ARG unused) {
  return val_a * val_b;
}

Tensor tensor_mul(const Tensor tensor_a, const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, 0, mul);
}

static inline float scale_fn(float UNUSED_ARG val_a, float val_b,
                             float scalar) {
  return val_b * scalar;
}
Tensor tensor_scale(const Tensor tensor, f32 scalar) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto val = tensor->data->data[tensor->offset + i];
      res->data->data[i] = scalar * val;
    }
  } else
    tensor_binary_op_inplace(res, tensor, scalar, scale_fn);

  return res;
}
