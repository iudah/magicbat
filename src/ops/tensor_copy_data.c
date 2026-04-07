#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"

bool tensor_copy_data(Tensor dst, const Tensor src) {
  if (!dst || !src || dst->data->nelements != src->data->nelements)
    return false;

  for (u32 i = 0; i < dst->data->nelements; i++) {
    dst->data->data[i] = src->data->data[i];
  }

  return true;
}