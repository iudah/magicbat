#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float mul(float a, float b) { return a * b; }

Tensor tensor_mul(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, mul);
}
