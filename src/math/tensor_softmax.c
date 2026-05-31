#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_softmax_axis(const Tensor t, i32 axis) {
  if (!t)
    return nullptr;

  Tensor res = nullptr;

  Tensor max = tensor_max_axis(t, axis);
  if (max == nullptr)
    return nullptr;

  Tensor z = tensor_sub(t, max);
  if (z == nullptr)
    goto free_max;

  Tensor expz = tensor_exp(z);
  if (expz == nullptr)
    goto free_z;

  Tensor sum = tensor_sum_axis(expz, axis);
  if (sum == nullptr)
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
    return nullptr;

  Tensor res = nullptr;

  Tensor max = tensor_new(1, (u32[]){1});
  if (max == nullptr)
    return nullptr;
  tensor_fill(max, tensor_max_all(t));

  Tensor z = tensor_sub(t, max);
  if (z == nullptr)
    goto free_max;

  Tensor expz = tensor_exp(z);
  if (expz == nullptr)
    goto free_z;

  Tensor sum = tensor_new(1, (u32[]){1});
  if (sum == nullptr)
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
