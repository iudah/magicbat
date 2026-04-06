#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_memory.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

extern thread_local u32 *oneptr;

bool tensor_destroy(Tensor t) {
  if (t == NULL)
    return false;

  if (t->data)
    tfree(t->data);
  if (t->shape && t->shape != oneptr)
    tfree(t->shape);

  memset(t, 0, sizeof(*t));

  tfree(t);

  return true;
}
