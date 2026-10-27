#ifndef ADAM_H
#define ADAM_H

#include "tensor.h"

typedef struct adam_optimizer *AdamOptimizer;

AdamOptimizer adam_optimizer_new(Tensor *parameters, u32 n_parameters,
                                 f32 learn_rate, f32 grad_clip,
                                 f32 global_norm_max, f32 beta1, f32 beta2,
                                 f32 step, f32 epsilon);
AdamOptimizer adamw_optimizer_new(Tensor *parameters, u32 n_parameters,
                                  f32 learn_rate, f32 grad_clip,
                                  f32 global_norm_max, f32 beta1, f32 beta2,
                                  f32 weight_decay, f32 step, f32 epsilon);
bool          adam_optimizer_destroy(AdamOptimizer optimizer);
bool          adam_optimize(AdamOptimizer adam, Tensor *grads);
bool          adamw_optimize(AdamOptimizer adam, Tensor *grads);

#endif
