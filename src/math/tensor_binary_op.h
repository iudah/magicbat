#ifndef TENSOR_BINARY_OP_H
#define TENSOR_BINARY_OP_H
#include "tensor.h"
#include <stdint.h>

Tensor tensor_binary_op(Tensor restrict tensor_a, Tensor restrict tensor_b,
                        f32 alpha,
                        float (*operation_callback)(float val_a, float val_b,
                                                    float alpha));

bool tensor_binary_op_inplace(
    Tensor restrict tensor_a, Tensor restrict tensor_b, f32 alpha,
    float (*operation_callback)(float val_a, float val_b, float alpha));

bool tensor_binary_op_scalar_inplace(Tensor restrict tensor_a, f32 scalar,
                                     float (*operation_callback)(float val_a,
                                                                 float val_b));

#endif
