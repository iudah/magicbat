#include "sgd.h"
#include "sgd_prot.h"
#include "tensor_memory.h"
#include <stdint.h>

bool sgd_optimizer_destroy(SgdOptimizer optimizer) {
  if (!optimizer)
    return false;

  if (optimizer->base.parameters)
    tfree(optimizer->base.parameters);

  optimizer->learn_rate = 0;

  return tfree(optimizer);
}
