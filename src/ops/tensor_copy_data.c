#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_odometer.h"
#include <string.h>

bool tensor_copy_data(Tensor dst, const Tensor src) {
  if (!dst || !src || dst->data == src->data || dst->ndims != src->ndims)
    return false;

  u32 dst_len = 1;
  bool use_memcpy = dst->is_contiguous && src->is_contiguous;
  for (u32 i = 0; i < dst->ndims; ++i) {
    if (dst->shape[i] != src->shape[i])
      return false;

    dst_len *= dst->shape[i];
  }

  if (use_memcpy) {
    memcpy(dst->data->data + dst->offset, src->data->data + src->offset,
           dst_len * sizeof(f32));
  } else {
    auto index = tensor_odometer_new(dst->ndims);
    do {
      auto value = tensor_get(src, index);
      tensor_set(dst, index, value);
    } while (tensor_odometer_next(index, dst->ndims, dst->shape));
    tensor_odometer_destroy(index);
  }

  return true;
}
