#ifndef TENSOR_PROT_H
#define TENSOR_PROT_H

#include "../tensor.h"
#include "type_alias.h"

struct tensor_struct {
  f32 *data;
  u32 *shape;
  u32 ndims;
  u32 nelements;
};

#endif
