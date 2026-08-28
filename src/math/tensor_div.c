#include "tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float divide(float val_a, float val_b, f32 UNUSED_ARG unused) {
  return val_a / val_b;
}

Tensor tensor_div(const Tensor tensor_a, const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, 0, divide);
}

static inline float divisor_backward(float dvsr, float grad,
                                     f32 UNUSED_ARG unused) {
  return grad / (dvsr * dvsr);
}

Tensor tensor_divisor_backward(const Tensor divisor, const Tensor grad) {
  return tensor_binary_op(divisor, grad, 0, divisor_backward);
}
