#include "tensor_binary_op.h"
#include "tensor_reduce.h"
#include <math.h>

#define EPS 1e-6

static inline f32 layer_norm_var(f32 accumulator, f32 curr, f32 alpha,
                                 bool is_last_in_axis) {
  accumulator += curr * curr;
  return !is_last_in_axis ? accumulator : ((accumulator / alpha) + EPS);
}

static inline float divide(float deviation, float variance,
                           float UNUSED_ARG arg) {
  return deviation / sqrtf(variance);
}

Tensor tensor_layer_norm_devation_variance_axis(Tensor deviation, u32 axis) {
  return tensor_reduce_axis(deviation, axis, 0, deviation->shape[axis],
                            layer_norm_var);
}

Tensor tensor_layer_norm_axis(Tensor tensor, u32 axis,
                              Tensor *variance_holder) {
  auto mean = tensor_mean_axis(tensor, axis);
  auto deviation = tensor_sub(tensor, mean);
  auto var = tensor_reduce_axis(deviation, axis, 0, deviation->shape[axis],
                                layer_norm_var);

  if (variance_holder)
    *variance_holder = var;

  auto res = tensor_binary_op(deviation, var, 0, divide);

  tensor_destroy(deviation);
  tensor_destroy(mean);

  return res;
}

static inline float divide_grad(float net_grad, float variance,
                                float UNUSED_ARG args) {
  return -net_grad / sqrtf(variance);
}
Tensor tensor_layer_norm_axis_backward(Tensor gradient, Tensor normal, u32 axis,
                                       Tensor variance) {
  auto mu_grad = tensor_mean_axis(gradient, axis);
  auto grad_x_norm = tensor_mul(gradient, normal);
  auto mu_dy_norm = tensor_mean_axis(grad_x_norm, axis);
  auto norm_x_mu_dy_norm = tensor_mul(normal, mu_dy_norm);
  tensor_add_inplace(norm_x_mu_dy_norm, mu_grad);
  auto net_grad = norm_x_mu_dy_norm;
  tensor_sub_inplace(norm_x_mu_dy_norm, gradient);

  auto grad = tensor_binary_op(net_grad, variance, 0, divide_grad);

  // tensor_destroy(net_grad);
  tensor_destroy(norm_x_mu_dy_norm);
  tensor_destroy(mu_dy_norm);
  tensor_destroy(grad_x_norm);
  tensor_destroy(mu_grad);

  return grad;
}
