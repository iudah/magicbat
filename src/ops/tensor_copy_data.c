#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_copy_data(Tensor dst, const Tensor src) {
  if (!dst || !src || dst->nelements != src->nelements)
    return false;

  for (u32 i = 0; i < dst->nelements; i++) {
    dst->data[i] = src->data[i];
  }

  return true;
}