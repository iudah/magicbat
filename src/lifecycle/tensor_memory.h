#ifndef TENSOR_MEMORY_H
#define TENSOR_MEMORY_H

#include <stdbool.h>
#include <stdlib.h>

#include "type_alias.h"

static inline mem tmalloc(u64 size) { return malloc(size); }
static inline mem tcalloc(u64 count, u64 size) { return calloc(count, size); }
static inline mem trealloc(mem ptr, u64 size) { return realloc(ptr, size); }
static inline bool tfree(mem addr) {
  free(addr);
  return true;
}

#endif
