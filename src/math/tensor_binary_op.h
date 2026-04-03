#ifndef TENSOR_BINARY_OP_H
#define TENSOR_BINARY_OP_H
#include "../../include/tensor/tensor.h"
#include <stdint.h>

Tensor tensor_binary_op(const Tensor t, const Tensor s,
                        float (*op)(float, float));

#endif
