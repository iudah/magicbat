#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float add(float a, float b) { return a + b; }

Tensor tensor_add(const Tensor t, const Tensor s) {
  return tensor_binary_op(t, s, add);
}

thread_local static float _alpha;
static inline float scaled_add(float a, float b) { return a + _alpha * b; }

Tensor tensor_scaled_add(const Tensor t, f32 alpha, const Tensor s) {
  _alpha = alpha;
  return tensor_binary_op(t, s, scaled_add);
}

bool tensor_add_inplace(Tensor t, Tensor s) {
  return tensor_binary_op_inplace(t, s, add);
}
