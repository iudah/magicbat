#ifndef TENSOR_BINARY_OP_H
#define TENSOR_BINARY_OP_H
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_binary_op(Tensor tensor_a, Tensor tensor_b,
                        float (*operation_callback)(float, float));

bool tensor_binary_op_inplace(Tensor restrict tensor_a,
                              Tensor restrict tensor_b,
                              float (*operation_callback)(float, float));

bool tensor_binary_op_scalar_inplace(Tensor restrict tensor_a, f32 scalar,
                                     float (*operation_callback)(float, float));
#endif
