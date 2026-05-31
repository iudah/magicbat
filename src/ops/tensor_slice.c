#include "tensor.h"
#include "tensor_memory.h"
#include "tensor_prot.h"
#include <stdatomic.h>

Tensor tensor_slice(Tensor tensor, TensorSlice *slices) {
  Tensor slice = tmalloc(sizeof(*slice));

  slice->ndims = tensor->ndims;
  slice->is_contiguous = false;
  slice->data = tensor->data;
  atomic_fetch_add(&slice->data->refcount, 1);

  slice->offset = tensor->offset;
  u32 steps[MAX_DIMS];
  for (u32 i = 0; i < slice->ndims; ++i) {
    steps[i] = (slices[i].step <= 1) ? 1 : slices[i].step;
  }

  for (u32 i = 0; i < slice->ndims; ++i) {
    slice->offset += slices[i].start * tensor->stride[i];
    slice->shape[i] = (slices[i].stop - slices[i].start) / steps[i];
    slice->stride[i] = tensor->stride[i] * steps[i];
  }

  return slice;
}
