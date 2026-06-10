#include "tensor_reduce.h"

f32 mean(f32 accumulator, f32 curr, f32 alpha, bool is_last_in_axis) {
  accumulator += curr;
  if (is_last_in_axis) {
    return accumulator / alpha;
  }
  return accumulator;
}

Tensor tensor_mean_axis(Tensor tensor, u32 axis) {
  return tensor_reduce_axis(tensor, axis, 0, tensor->shape[axis], mean);
}
