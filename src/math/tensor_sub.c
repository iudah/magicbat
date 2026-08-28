#include "tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float subtract(float val_a, float val_b, float UNUSED_ARG alpha) {
  return val_a - val_b;
}

Tensor tensor_sub(const Tensor tensor_a, const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, 0, subtract);
}

bool tensor_sub_inplace(Tensor tensor_a, Tensor tensor_b) {
  return tensor_binary_op_inplace(tensor_a, tensor_b, 0, subtract);
}
