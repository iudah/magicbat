#include "tensor_prot.h"
#include "tensor.h"
#include "data_storage.h"
#include "tensor_memory.h"
#include <stdint.h>

Tensor tensor_scalar(f32 value) {
  Tensor tensor = tmalloc(sizeof(*tensor));
  if (tensor == nullptr)
    return nullptr;

  tensor->ndims = 1;

  tensor->data = data_storage_new(1);

  *tensor->data->data = value;
  tensor->shape[0] = 1;
  tensor->stride[0] = 0;

  tensor->refcount = 1;
  tensor->requires_grad = false;
  tensor->is_tensor_type = true;
  tensor->is_contiguous = true;
  tensor->offset = 0;

  return tensor;
}
