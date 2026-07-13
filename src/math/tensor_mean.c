#include "tensor_reduce.h"

f32 mean(f32 accumulator, f32 curr, f32 alpha, bool is_last_in_axis) {
  accumulator += curr;
  return !is_last_in_axis ? accumulator : accumulator / alpha;
}

Tensor tensor_mean_axis(Tensor tensor, u32 axis) {
  return tensor_reduce_axis(tensor, axis, 0, tensor->shape[axis], mean);
}

f32 tensor_mean_all(Tensor tensor) {
  return tensor_reduce_all(tensor, 0, tensor_num_elements(tensor), mean);
}
