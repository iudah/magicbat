#include "sgd.h"
#include "sgd_prot.h"
#include "var.h"
#include <stdint.h>

bool sgd_optimize(SgdOptimizer optimizer, Tensor *grads) {
  if (!optimizer)
    return false;

  if (!grads) {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = var_grad(optimizer->parameters[i]);
      auto param = optimizer->parameters[i];
      if (!grad)
        continue;

      auto new_param = tensor_scaled_add(param, -optimizer->learn_rate, grad);
      tensor_copy_data(param, new_param);

      tensor_destroy(new_param);
    }
  } else {
    for (u32 i = 0; i < optimizer->n_parameters; ++i) {
      auto grad = grads[i];
      auto param = optimizer->parameters[i];
      if (!grad)
        continue;

      auto new_param = tensor_scaled_add(param, -optimizer->learn_rate, grad);
      tensor_copy_data(param, new_param);

      tensor_destroy(new_param);
    }
  }

  return true;
}
