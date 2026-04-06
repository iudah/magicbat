#ifndef TENSOR_INCREMENT_INDEX_H
#define TENSOR_INCREMENT_INDEX_H

#include "../../include/adt/type_alias.h"

static inline void tensor_increment_index(u32 ndims, u32 *index,
                                          const u32 *shape) {
  for (u32 i = ndims; i > 0;) {
    i--;
    index[i]++;
    if (index[i] < shape[i]) {
      break;
    }
    index[i] = 0;
  }
}

#endif