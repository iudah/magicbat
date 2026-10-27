#ifndef SGD_PROT_H
#define SGD_PROT_H

#include "../../tensor.h"

struct optimizer_base {
  Tensor *parameters;
  f32 max_norm;
  f32 clip_threshold;
  u32 n_parameters;
};

typedef struct optimizer_base *Optimizer;

struct sgd_optimizer {
  struct optimizer_base base;
  float learn_rate;
};

#endif
