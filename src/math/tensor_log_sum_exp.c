#include "tensor.h"
#include <stdint.h>

Tensor tensor_log_sum_exp_axis(const Tensor tensor, i32 axis) {
  if (!tensor)
    return nullptr;

  Tensor res = nullptr;

  Tensor max = tensor_max_axis(tensor, axis);
  if (max == nullptr)
    return nullptr;

  Tensor normalised = tensor_sub(tensor, max);
  if (normalised == nullptr)
    goto free_max;

  Tensor expz = tensor_exp(normalised);
  if (expz == nullptr)
    goto free_z;

  Tensor sum = tensor_sum_axis(expz, axis);
  if (sum == nullptr)
    goto free_expz;

  Tensor log = tensor_log(sum);
  if (!log)
    goto free_sum;

  res = tensor_add(log, max);

  tensor_destroy(log);
free_sum:
  tensor_destroy(sum);
free_expz:
  tensor_destroy(expz);
free_z:
  tensor_destroy(normalised);
free_max:
  tensor_destroy(max);

  return res;
}

Tensor tensor_log_sum_exp(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = nullptr;

  Tensor max = tensor_new(1, (u32[]){1});
  if (max == nullptr)
    return nullptr;
  tensor_fill(max, tensor_max(tensor));

  Tensor normalised = tensor_sub(tensor, max);
  if (normalised == nullptr)
    goto free_max;

  Tensor expz = tensor_exp(normalised);
  if (expz == nullptr)
    goto free_z;

  Tensor sum = tensor_new(1, (u32[]){1});
  if (sum == nullptr)
    goto free_expz;
  tensor_fill(sum, tensor_sum(expz));

  Tensor logz = tensor_log(sum);
  if (!logz)
    goto free_sum;

  res = tensor_add(logz, max);

  tensor_destroy(logz);
free_sum:
  tensor_destroy(sum);
free_expz:
  tensor_destroy(expz);
free_z:
  tensor_destroy(normalised);
free_max:
  tensor_destroy(max);

  return res;
}
