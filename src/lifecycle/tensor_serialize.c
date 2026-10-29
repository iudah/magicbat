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
    // NOTE: weird stuff, why am i writing strides when still resolve then
    // non-contiguous to be contiguous (2nd branch)?
    fwrite(tensor->stride, sizeof(tensor->stride), 1, binary);
    data_storage_write_offset(tensor->data, tensor->offset, binary);
  } else {
    // NOTE: maybe i'll remove this later. use a condition to make a new tensor
    // of contigency and write after the condition.
    auto tmp = tensor_to_contiguous(tensor);
    fwrite(tmp->stride, sizeof(tmp->stride), 1, binary);
    data_storage_write_offset(tmp->data, tmp->offset, binary);
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
  tensor->data = data_storage_read(binary);
  if (!tensor->data || !tensor->data->data) {
    tfree(tensor);
    return nullptr;
  }
  tensor->requires_grad  = false;
  tensor->refcount       = 1;
  tensor->is_tensor_type = true;
  tensor->is_contiguous  = true;
  tensor->offset         = 0;

  TASSERT(tensor->data && tensor->data->data);

  return tensor;
}
