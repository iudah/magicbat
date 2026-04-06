#include "../../include/tensor.h"
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

Tensor tensor_softmax_all(const Tensor t) {
  if (!t)
    return NULL;

  Tensor res = NULL;

  Tensor max = tensor_new(1, (u32[]){1});
  if (max == NULL)
    return NULL;
  tensor_fill(max, tensor_max_all(t));

  Tensor z = tensor_sub(t, max);
  if (z == NULL)
    goto free_max;

  Tensor expz = tensor_exp(z);
  if (expz == NULL)
    goto free_z;

  Tensor sum = tensor_new(1, (u32[]){1});
  if (sum == NULL)
    goto free_expz;
  tensor_fill(sum, tensor_sum_all(expz));

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
