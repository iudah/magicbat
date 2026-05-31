#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "data_storage.h"
#include "tensor_memory.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool tensor_destroy(Tensor tensor) {
  if (tensor == nullptr)
    return false;

  if (!tensor->is_tensor_type)
    return false;

  if (atomic_fetch_sub(&tensor->refcount, 1) != 1)
    return true;

  if (tensor->data)
    data_storage_destroy(tensor->data);

  tfree(tensor);

  return true;
}
