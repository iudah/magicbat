#include "sgd.h"
#include "sgd_prot.h"
#include "tensor_memory.h"
#include <stdint.h>
#include <string.h>

SgdOptimizer sgd_optimizer_new(Tensor *parameters, u32 n_parameters,
                               f32 learn_rate, f32 grad_clip,
                               f32 global_norm_max) {
  SgdOptimizer optimizer = tmalloc(sizeof(*optimizer));
  if (!optimizer)
    return nullptr;

  auto tmp = tmalloc(sizeof(Tensor) * n_parameters);
  if (!tmp) {
    tfree(optimizer);
    return nullptr;
  }

  optimizer->base.parameters = tmp;
  memcpy(optimizer->base.parameters, parameters, sizeof(Tensor) * n_parameters);
  optimizer->learn_rate = learn_rate;
  optimizer->base.n_parameters = n_parameters;
  optimizer->base.clip_threshold = grad_clip;
  optimizer->base.max_norm = global_norm_max;

  return optimizer;
}
