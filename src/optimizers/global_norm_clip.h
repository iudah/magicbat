#ifndef GLOBAL_NORM_CLIP
#define GLOBAL_NORM_CLIP

#include "sgd_prot.h"
#include "tensor.h"
#include "type_alias.h"
#include "var.h"
#include <math.h>

bool global_norm_clip(Optimizer optimizer, Tensor *grads, f32 *alpha) {
  if (!optimizer)
    return false;

  if (optimizer->max_norm < EPS) {
    *alpha = 1;
    return true;
  }

  f32 total_norm_sqr = 0.0F;
  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = var_grad(optimizer->parameters[i]);
      if (!grad)
        continue;

      total_norm_sqr += tensor_total_norm_sqr(grad);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = grads[i];
      if (!grad)
        continue;

      total_norm_sqr += tensor_total_norm_sqr(grad);
    }
  }

  total_norm_sqr = sqrtf(total_norm_sqr);
  *alpha         = optimizer->max_norm < total_norm_sqr
                       ? (optimizer->max_norm / (total_norm_sqr + EPS))
                       : 1.0F;

  return true;
}

#endif
