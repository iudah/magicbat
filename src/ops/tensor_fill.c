#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_fill(const Tensor t, f32 value) {
  if (t == NULL)
    return false;

  for (u32 i = 0; i < t->nelements; i++) {
    t->data[i] = value;
  }

  return true;
}