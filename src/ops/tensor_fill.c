#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_fill(const Tensor t, f32 value) {
  if (t == nullptr)
    return false;

  for (u32 i = 0; i < t->data->nelements; i++) {
    t->data->data[i] = value;
  }

  return true;
}
