#include "../../include/tensor.h"
#include "tensor_binary_op.h"
#include <stdint.h>

static inline float add(float float_a, float float_b,
                        float __attribute__((unused)) alpha) {
  return float_a + float_b;
}

static inline float add2(float float_a, float float_b) {
  return float_a + float_b;
}

Tensor tensor_add(const Tensor tensor_a, const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, 0, add);
}

static inline float scaled_add(float float_a, float float_b, f32 scale_alpha) {
  return float_a + (scale_alpha * float_b);
}

Tensor tensor_scaled_add(const Tensor tensor_a, f32 alpha,
                         const Tensor tensor_b) {
  return tensor_binary_op(tensor_a, tensor_b, alpha, scaled_add);
}

bool tensor_add_inplace(Tensor tensor_a, Tensor tensor_b) {
  return tensor_binary_op_inplace(tensor_a, tensor_b, 0, add);
}

bool tensor_add_scalar_inplace(Tensor tensor_a, const f32 f_value) {
  return tensor_binary_op_scalar_inplace(tensor_a, f_value, add2);
}
