#include "global_norm_clip.h"
#include "grad_clamp.h"
#include "sgd.h"
#include "sgd_prot.h"
#include "tensor.h"
#include "var.h"
#include <stdint.h>

bool sgd_optimize(SgdOptimizer sgd, Tensor *grads) {
  if (!sgd)
    return false;

  auto learn_rate = -sgd->learn_rate;
  f32 alpha = 1;
  if (!global_norm_clip(&sgd->base, grads, &alpha))
    return false;
  learn_rate *= alpha;

  if (!gradient_clamp(&sgd->base, grads, alpha))
    return false;

  auto optimizer = &sgd->base;
  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto param = optimizer->parameters[i];
      auto grad = var_grad(param);
      if (!grad)
        continue;

      auto new_param = tensor_scaled_add(param, learn_rate, grad);
      tensor_copy_data(param, new_param);

      tensor_destroy(new_param);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = grads[i];
      auto param = optimizer->parameters[i];
      if (!grad)
        continue;

      auto new_param = tensor_scaled_add(param, learn_rate, grad);
      tensor_copy_data(param, new_param);

      tensor_destroy(new_param);
    }
  }

  return true;
}
