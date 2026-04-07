#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_matmul(const Tensor t, const Tensor s) {
  if (!t || !s)
    return NULL;
  if (s->ndims != 2 || t->ndims != 2 || t->shape[1] != s->shape[0])
    return NULL;

  u32 row = t->shape[0];
  u32 col = s->shape[1];
  u32 com = t->shape[1];

  // guarantees zeros
  Tensor res = tensor_new(2, (u32[]){row, col});
  if (res == NULL)
    return NULL;

  for (u32 i = 0; i < row; ++i) {
    float *t_data = &t->data->data[i * com];
    float *r_data = &res->data->data[i * col];
    for (u32 j = 0; j < com; ++j) {
      float *s_data = &s->data->data[j * col];
      float t_val = t_data[j];
      for (u32 k = 0; k < col; ++k) {
        r_data[k] += t_val * s_data[k];
      }
    }
  }

  return res;
}
