#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float divide(float a, float b) { return a / b; }

Tensor tensor_div(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, divide);
}
