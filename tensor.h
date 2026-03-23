#ifndef TENSOR_H
#define TENSOR_H

#include "adt/type_alias.h"
#include <stdbool.h>

typedef struct tensor_struct *Tensor;

Tensor tensor_new(u32 ndims, u32 *shape);
bool tensor_destroy(Tensor t);

#endif
