#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include <stdint.h>

u32 one = 1;
thread_local u32 *oneptr = &one;

Tensor tensor_scalar(f32 value) {
  Tensor t = tmalloc(sizeof(*t));
  if (t == NULL)
    return NULL;

  t->ndims = 1;

  t->data = tmalloc(sizeof(f32));
  *t->data = value;

  t->nelements = 1;
  t->shape = oneptr;

  return t;
}
