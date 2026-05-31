#ifndef TENSOR_PROT_H
#define TENSOR_PROT_H

#include "type_alias.h"

#include <stdatomic.h>

#define MAX_DIMS 4

typedef struct data_storage data_storage;

struct data_storage {
  f32 *data;
  u32 nelements;
  _Atomic u32 refcount;
};

struct tensor_struct {
  data_storage *data;
  u32 shape[MAX_DIMS];
  u32 stride[MAX_DIMS];
  u32 offset;
  u32 ndims;
  _Atomic u32 refcount;
  bool requires_grad;
  bool is_tensor_type;
  bool is_contiguous;
};

#endif
