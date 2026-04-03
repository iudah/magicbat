#include "../../include/tensor/tensor.h"
#include <stdint.h>

Tensor tensor_softmax_axis(const Tensor t, i32 axis) {
  if (!t)
    return NULL;

  Tensor res = NULL;

  Tensor max = tensor_max_axis(t, axis);
  if (max == NULL)
    return NULL;

  Tensor z = tensor_sub(t, max);
  if (z == NULL)
    goto free_max;

  Tensor expz = tensor_exp(z);
  if (expz == NULL)
    goto free_z;

  Tensor sum = tensor_sum_axis(expz, axis);
  if (sum == NULL)
    goto free_expz;

  res = tensor_div(expz, sum);

  tensor_destroy(sum);
free_expz:
  tensor_destroy(expz);
free_z:
  tensor_destroy(z);
free_max:
  tensor_destroy(max);

  return res;
}
