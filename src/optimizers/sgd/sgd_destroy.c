#include "sgd_prot.h"
#include "sgd.h"
#include "tensor_memory.h"
#include <stdint.h>

bool sgd_optimizer_destroy(SgdOptimizer optimizer) {
  if (!optimizer)
    return false;

  if (optimizer->parameters)
    tfree(optimizer->parameters);

  optimizer->learn_rate = 0;

  return tfree(optimizer);
}
