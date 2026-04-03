#include "../../include/tensor/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float add(float a, float b) { return a + b; }

Tensor tensor_add(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, add);
}
