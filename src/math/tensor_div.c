#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float divide(float a, float b) { return a / b; }

Tensor tensor_div(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, divide);
}

static inline float divisor_backward(float dvsr, float grad) {
  return grad / (dvsr * dvsr);
}

Tensor tensor_divisor_backward(const Tensor tensor_a, const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, divisor_backward);
}
