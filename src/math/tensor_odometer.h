#ifndef TENSOR_ODOMETER_H
#define TENSOR_ODOMETER_H
#include "../../include/adt/type_alias.h"
#include "../../src/lifecycle/tensor_memory.h"
#include <stdint.h>

static inline mem tensor_odometer_new(u32 ndims) {
  return tcalloc(ndims, sizeof(u32));
}

static inline bool tensor_odometer_next(u32 *index, u32 ndims,
                                        const u32 *shape) {
  if (index == NULL)
    return false;

  u32 dim = ndims;
  while (dim > 0) {
    dim--;
    u32 i = ++(index)[dim];
    if (i == shape[dim]) {
      (index)[dim] = 0;
    } else {
      return true;
    }
  }
  return false;
}

static inline bool tensor_odometer_destroy(mem ptr) { return tfree(ptr); }
#endif