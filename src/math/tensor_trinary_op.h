#ifndef TENSOR_TRINARY_OP_H
#define TENSOR_TRINARY_OP_H
#include "tensor.h"
#include <stdint.h>

Tensor tensor_trinary_op(Tensor restrict tensor_a, Tensor restrict tensor_b,
                         Tensor restrict tensor_c,
                         float (*operation_callback)(float val_a, float val_b,
                                                     float alpha));

#endif
