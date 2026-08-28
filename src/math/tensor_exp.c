#include "tensor.h"
#include "tensor_binary_op.h"
#include "tensor_prot.h"
#include <math.h>
#include <stdint.h>

static inline float exp_fn(float UNUSED_ARG val_a, float val_b,
                           float UNUSED_ARG scalar) {
  return expf(val_b);
}
Tensor tensor_exp(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto val = tensor->data->data[i];
      res->data->data[i] = expf(val);
    }
  } else
    tensor_binary_op_inplace(res, tensor, 0, exp_fn);

  return res;
}
