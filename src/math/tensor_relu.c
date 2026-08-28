#include "tensor_prot.h"
#include "tensor.h"
#include "tensor_binary_op.h"
#include <math.h>
#include <stdint.h>

static inline float relu_fn(float UNUSED_ARG val_a, float val_b,
                            float UNUSED_ARG scalar) {
  return fmaxf(val_b, 0);
}
Tensor tensor_relu(const Tensor tensor) {
  TASSERT(tensor && "Null tensor");
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto value = tensor->data->data[i];
      res->data->data[i] = fmaxf(value, 0);
    }
  } else
    tensor_binary_op_inplace(res, tensor, 0, relu_fn);

  return res;
}

#define EPS (1e-8)
Tensor tensor_relu_backward(const Tensor tensor, const Tensor relu,
                            const Tensor grad) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(relu), tensor_shape(relu));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (grad) {
    for (u32 i = 0; i < nelement; ++i) {
      res->data->data[i] =
          relu->data->data[i] > EPS ? grad->data->data[i] : 0.0F;
    }
  } else {
    for (u32 i = 0; i < nelement; ++i) {
      auto value = relu->data->data[i];
      res->data->data[i] = value > EPS ? 1.0F : 0.0F;
    }
  }

  return res;
}
