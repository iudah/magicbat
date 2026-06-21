#include "tensor_binary_op.h"
#include "tensor_reduce.h"

f32 mean_sqr(f32 accumulator, f32 curr, f32 alpha, bool is_last_in_axis) {
  accumulator += curr * curr;
  return !is_last_in_axis ? accumulator : (accumulator / alpha);
}

Tensor tensor_mean_sqr_axis(Tensor tensor, u32 axis) {
  return tensor_reduce_axis(tensor, axis, 0, tensor->shape[axis], mean_sqr);
}

f32 tensor_mean_sqr_all(Tensor tensor) {
  u32 len = 1;
  for (u32 i = 0; i < tensor->ndims; ++i) {
    len *= tensor->shape[i];
  }
  return tensor_reduce_all(tensor, 0, len, mean_sqr);
}

static inline float d_ms(float grad, float val, float length) {
  return grad * val * 2 / length;
}
static inline float d_ms_no_grad(float val, float UNUSED_ARG unused,
                                 float length) {
  return val * 2 / length;
}
Tensor tensor_mean_sqr_axis_backward(Tensor gradient, Tensor tensor, u32 axis) {
  if (gradient) {
    return tensor_binary_op(gradient, tensor, (f32)tensor->shape[axis], d_ms);
  }
  return tensor_binary_op(tensor, tensor, (f32)tensor->shape[axis],
                          d_ms_no_grad);
}

static inline float d_ms_no_grad_flat(float val, float UNUSED_ARG unused,
                                      float alpha) {
  return val * alpha;
}
Tensor tensor_mean_sqr_all_backward(f32 gradient, Tensor tensor) {
  u32 len = 1;
  for (u32 i = 0; i < tensor->ndims; ++i) {
    len *= tensor->shape[i];
  }
  return tensor_binary_op(tensor, tensor, 2 * gradient / (f32)len,
                          d_ms_no_grad_flat);
}
