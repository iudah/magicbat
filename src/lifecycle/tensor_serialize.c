#include "data_storage.h"
#include "tensor.h"
#include "tensor_prot.h"
#include <stdio.h>

bool tensor_write(Tensor tensor, FILE *binary) {
  if (tensor == nullptr) {
    return false;
  }

  fwrite(&tensor->ndims, sizeof(tensor->ndims), 1, binary);
  fwrite(tensor->shape, sizeof(tensor->shape), 1, binary);
  if (tensor->is_contiguous) {
    fwrite(tensor->stride, sizeof(tensor->stride), 1, binary);
    data_storage_write(tensor->data, binary);
  } else {
    auto tmp = tensor_to_contiguous(tensor);
    fwrite(tmp->stride, sizeof(tmp->stride), 1, binary);
    data_storage_write(tmp->data, binary);
    tensor_destroy(tmp);
  }

  return true;
}

Tensor tensor_read(FILE *binary) {
  Tensor tensor = tcalloc(1, sizeof(*tensor));
  if (tensor == nullptr) {
    return nullptr;
  }

  fread(&tensor->ndims, sizeof(tensor->ndims), 1, binary);
  fread(tensor->shape, sizeof(tensor->shape), 1, binary);
  fread(tensor->stride, sizeof(tensor->stride), 1, binary);
  tensor->data           = data_storage_read(binary);
  tensor->requires_grad  = false;
  tensor->refcount       = 1;
  tensor->is_tensor_type = true;
  tensor->is_contiguous  = true;
  tensor->offset         = 0;

  TASSERT(tensor->data && tensor->data->data);

  return tensor;
}
