#include "adam.h"
#include "global_norm_clip.h"
#include "grad_clamp.h"
#include "sgd_prot.h"
#include "tensor.h"
#include "tensor_binary_op.h"
#include "tensor_memory.h"
#include "tensor_trinary_op.h"
#include "type_alias.h"
#include <math.h>

struct adam_optimizer {
  struct optimizer_base base;

  f32 learn_rate;
  f32 beta1;
  f32 beta2;
  f32 beta1_prev;
  f32 beta2_prev;
  f32 epsilon;
  f32 weight_decay;

  struct adam_slot *slots;
};
struct adam_slot {
  Tensor m;
  Tensor v;
};

AdamOptimizer adam_optimizer_new(Tensor *parameters, u32 n_parameters,
                                 f32 learn_rate, f32 grad_clip,
                                 f32 global_norm_max, f32 beta1, f32 beta2,
                                 f32 step, f32 epsilon) {
  return adamw_optimizer_new(parameters, n_parameters, learn_rate, grad_clip,
                             global_norm_max, beta1, beta2, 0.0F, step,
                             epsilon);
}
AdamOptimizer adamw_optimizer_new(Tensor *parameters, u32 n_parameters,
                                  f32 learn_rate, f32 grad_clip,
                                  f32 global_norm_max, f32 beta1, f32 beta2,
                                  f32 weight_decay, f32 step, f32 epsilon) {

  AdamOptimizer optimizer = tmalloc(sizeof(*optimizer));
  if (!optimizer)
    return nullptr;

  auto tmp = tmalloc(sizeof(Tensor) * n_parameters);
  if (!tmp) {
    tfree(optimizer);
    return nullptr;
  }
  auto tmp1 = tcalloc(n_parameters, sizeof(*optimizer->slots));
  if (!tmp1) {
    tfree(tmp);
    tfree(optimizer);
    return nullptr;
  }

  optimizer->base.n_parameters   = n_parameters;
  optimizer->base.clip_threshold = grad_clip;
  optimizer->base.max_norm       = global_norm_max;
  optimizer->base.parameters     = tmp;
  memcpy(optimizer->base.parameters, parameters, sizeof(Tensor) * n_parameters);

  optimizer->learn_rate   = learn_rate;
  optimizer->beta1        = beta1;
  optimizer->beta2        = beta2;
  optimizer->beta1_prev   = powf(beta1, step);
  optimizer->beta2_prev   = powf(beta2, step);
  optimizer->epsilon      = epsilon;
  optimizer->weight_decay = weight_decay;
  optimizer->slots        = tmp1;

  for (u32 i = 0; i < n_parameters; ++i) {
    optimizer->slots[i].m =
        tensor_new(tensor_ndims(parameters[i]), tensor_shape(parameters[i]));
    optimizer->slots[i].v =
        tensor_new(tensor_ndims(parameters[i]), tensor_shape(parameters[i]));
  }

  return optimizer;
}

bool adam_optimizer_destroy(AdamOptimizer optimizer) {
  if (!optimizer)
    return false;

  if (optimizer->base.parameters)
    tfree(optimizer->base.parameters);

  if (optimizer->slots) {
    for (u32 i = 0; i < optimizer->base.n_parameters; ++i) {
      tensor_destroy(optimizer->slots[i].m);
      tensor_destroy(optimizer->slots[i].v);
    }
    tfree(optimizer->slots);
  }

  optimizer->learn_rate = 0;
  tfree(optimizer);
  return true;
}

static inline float update_m(float float_a, float float_b, f32 beta) {
  return (beta * float_a) + ((1 - beta) * float_b);
}
static bool tensor_update_adam_m(Tensor adam_m, f32 beta1, Tensor grad) {
  return tensor_binary_op_inplace(adam_m, grad, beta1, update_m);
}
static inline float update_v(float float_a, float float_b, f32 beta) {
  return (beta * float_a) + ((1 - beta) * float_b * float_b);
}
static bool tensor_update_adam_v(Tensor adam_v, f32 beta2, Tensor grad) {
  return tensor_binary_op_inplace(adam_v, grad, beta2, update_v);
}
struct adam_tri_state {
  f32 lr_over_bc1;
  f32 inv_bc2;
  f32 epsilon;
  f32 weight_decay;
  f32 scaled_lr;
};

static float adam_opt_cb(float param, float adam_m, f32 adam_v, void *state) {
  struct adam_tri_state *adam     = state;
  float                  m_hat    = adam_m * adam->lr_over_bc1;
  float                  v_hat    = adam_v * adam->inv_bc2;
  float                  adaptive = m_hat / (sqrtf(v_hat) + adam->epsilon);
  return param - adaptive;
}
static float adamw_opt_cb(float param, float adam_m, f32 adam_v, void *state) {
  struct adam_tri_state *adam     = state;
  float                  m_hat    = adam_m * adam->lr_over_bc1;
  float                  v_hat    = adam_v * adam->inv_bc2;
  float                  adaptive = m_hat / (sqrtf(v_hat) + adam->epsilon);
  float                  decay = adam->scaled_lr * adam->weight_decay * param;
  return param - (adaptive + decay);
}
bool adam_optimize(AdamOptimizer adam, Tensor *grads) {
  f32 alpha = 1;
  if (!global_norm_clip(&adam->base, grads, &alpha))
    return false;

  if (!gradient_clamp(&adam->base, grads, alpha))
    return false;

  auto optimizer    = &adam->base;
  auto weight_decay = adam->weight_decay;
  auto beta1        = adam->beta1;
  auto beta2        = adam->beta2;
  adam->beta1_prev *= beta1;
  adam->beta2_prev *= beta2;
  auto       bias_corr_1 = 1 - adam->beta1_prev;
  auto       bias_corr_2 = 1 - adam->beta2_prev;
  const bool has_decay   = adam->weight_decay > EPS;

  struct adam_tri_state state = {
      .lr_over_bc1  = adam->learn_rate * alpha / bias_corr_1,
      .inv_bc2      = 1.0F / bias_corr_2,
      .epsilon      = adam->epsilon,
      .weight_decay = adam->weight_decay,
      .scaled_lr    = adam->learn_rate * alpha,
  };

  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto param = optimizer->parameters[i];
      auto grad  = var_grad(param);
      if (!grad)
        continue;

      auto adam_g =
          has_decay ? tensor_scaled_add(grad, adam->weight_decay, param) : grad;
      tensor_update_adam_m(adam->slots[i].m, beta1, adam_g);
      tensor_update_adam_v(adam->slots[i].v, beta2, adam_g);
      if (has_decay)
        tensor_destroy(adam_g);
      Tensor new_param = tensor_trinary_op(
          param, adam->slots[i].m, adam->slots[i].v, &state, adam_opt_cb);
      if (!new_param)
        return false;
      tensor_copy_data(param, new_param);
      tensor_destroy(new_param);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad  = grads[i];
      auto param = optimizer->parameters[i];
      if (!grad)
        continue;

      auto adam_g =
          has_decay ? tensor_scaled_add(grad, adam->weight_decay, param) : grad;
      tensor_update_adam_m(adam->slots[i].m, beta1, adam_g);
      tensor_update_adam_v(adam->slots[i].v, beta2, adam_g);
      if (has_decay)
        tensor_destroy(adam_g);
      Tensor new_param = tensor_trinary_op(
          param, adam->slots[i].m, adam->slots[i].v, &state, adam_opt_cb);
      if (!new_param)
        return false;
      tensor_copy_data(param, new_param);
      tensor_destroy(new_param);
    }
  }
  return true;
}

bool adamw_optimize(AdamOptimizer adam, Tensor *grads) {
  f32 alpha = 1;
  if (!global_norm_clip(&adam->base, grads, &alpha))
    return false;

  if (!gradient_clamp(&adam->base, grads, alpha))
    return false;

  auto optimizer = &adam->base;
  auto beta1     = adam->beta1;
  auto beta2     = adam->beta2;
  adam->beta1_prev *= beta1;
  adam->beta2_prev *= beta2;
  auto bias_corr_1 = 1 - adam->beta1_prev;
  auto bias_corr_2 = 1 - adam->beta2_prev;

  struct adam_tri_state state = {
      .lr_over_bc1  = adam->learn_rate * alpha / bias_corr_1,
      .inv_bc2      = 1.0F / bias_corr_2,
      .epsilon      = adam->epsilon,
      .weight_decay = adam->weight_decay,
      .scaled_lr    = adam->learn_rate * alpha,
  };

  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto param = optimizer->parameters[i];
      auto grad  = var_grad(param);
      if (!grad)
        continue;

      auto adam_g = grad;
      tensor_update_adam_m(adam->slots[i].m, beta1, adam_g);
      tensor_update_adam_v(adam->slots[i].v, beta2, adam_g);
      Tensor new_param = tensor_trinary_op(
          param, adam->slots[i].m, adam->slots[i].v, &state, adamw_opt_cb);
      if (!new_param)
        return false;
      tensor_copy_data(param, new_param);
      tensor_destroy(new_param);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad  = grads[i];
      auto param = optimizer->parameters[i];
      if (!grad)
        continue;

      auto adam_g = grad;
      tensor_update_adam_m(adam->slots[i].m, beta1, adam_g);
      tensor_update_adam_v(adam->slots[i].v, beta2, adam_g);
      Tensor new_param = tensor_trinary_op(
          param, adam->slots[i].m, adam->slots[i].v, &state, adamw_opt_cb);
      if (!new_param)
        return false;
      tensor_copy_data(param, new_param);
      tensor_destroy(new_param);
    }
  }
  return true;
}
