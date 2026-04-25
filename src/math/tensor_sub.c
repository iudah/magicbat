#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float subtract(float a, float b) { return a - b; }

Tensor tensor_sub(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, subtract);
}

bool tensor_sub_inplace(Tensor t, Tensor s) {
  return tensor_binary_op_inplace(t, s, subtract);
}
