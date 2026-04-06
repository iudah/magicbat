#ifndef SGD_PROT_H
#define SGD_PROT_H

#include "../../tensor.h"

struct sgd_optimizer {
  Tensor *parameters;
  float learn_rate;
  u32 n_parameters;
};

#endif
