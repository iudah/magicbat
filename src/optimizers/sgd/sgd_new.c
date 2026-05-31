#include "../../../include/adt/optimizers/sgd_prot.h"
#include "../../../include/optimizers/sgd.h"
#include "../../lifecycle/tensor_memory.h"
#include <stdint.h>
#include <string.h>

SgdOptimizer sgd_optimizer_new(Tensor *parameters, u32 n_parameters,
                               f32 learn_rate) {
  SgdOptimizer optimizer = tmalloc(sizeof(*optimizer));
  if (!optimizer)
    return nullptr;

  auto tmp = tmalloc(sizeof(Tensor) * n_parameters);
  if (!tmp) {
    tfree(optimizer);
    return nullptr;
  }

  optimizer->parameters = tmp;
  memcpy(optimizer->parameters, parameters, sizeof(Tensor) * n_parameters);
  optimizer->learn_rate = learn_rate;
  optimizer->n_parameters = n_parameters;

  return optimizer;
}
