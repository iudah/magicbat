#include "tensor.h"
#include "tensor_prot.h"
#include <stdio.h>

Tensor tensor_slice(Tensor tensor, TensorSlice *slices) {
  Tensor slice = tensor_view(tensor);
  slice->is_contiguous = false;

  u32 steps[MAX_DIMS];
  for (u32 i = 0; i < slice->ndims; ++i) {
    if (!(0 <= slices[i].start && slices[i].start < tensor->shape[i] &&
          slices[i].start < slices[i].stop &&
          slices[i].stop <= tensor->shape[i])) {
      fprintf(stderr, "Slice start or end is out of bound.\n");
      return nullptr;
    }
    steps[i] = (slices[i].step <= 1) ? 1 : slices[i].step;
  }

  for (u32 i = 0; i < slice->ndims; ++i) {
    slice->offset += slices[i].start * tensor->stride[i];
    slice->shape[i] =
        (slices[i].stop - slices[i].start + steps[i] - 1) / steps[i];
    slice->stride[i] = tensor->stride[i] * steps[i];
  }

  return slice;
}
