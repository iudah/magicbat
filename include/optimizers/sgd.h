#ifndef SGD_H
#define SGD_H

#include "tensor.h"

typedef struct sgd_optimizer *SgdOptimizer;

SgdOptimizer sgd_optimizer_new(Tensor *parameters, u32 n_parameters,
                               f32 learn_rate);
bool sgd_optimizer_destroy(SgdOptimizer optimizer);
bool sgd_optimize(SgdOptimizer optimizer, Tensor *grads);

#endif
