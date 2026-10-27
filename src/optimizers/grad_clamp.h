#ifndef GRADIENT_CLAMP
#define GRADIENT_CLAMP

#include "sgd_prot.h"
#include "tensor.h"
#include "type_alias.h"
#include "var.h"

#define EPS (1e-6)

bool gradient_clamp(Optimizer optimizer, Tensor *grads, f32 alpha) {
  if (!optimizer)
    return false;

  if (optimizer->clip_threshold < EPS) {
    return true;
  }

  auto clamp = optimizer->clip_threshold / alpha;

  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = var_grad(optimizer->parameters[i]);
      if (!grad)
        continue;

      tensor_clamp_inplace(grad, -clamp, clamp);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = grads[i];
      if (!grad)
        continue;

      tensor_clamp_inplace(grad, -clamp, clamp);
    }
  }

  return true;
}

#endif
