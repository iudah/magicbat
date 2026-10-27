#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include <stdint.h>

static inline f32 clamp(f32 val, f32 low, f32 high) {
  if (val < low)
    val = low;
  else if (val > high)
    val = high;

  return val;
}

void tensor_clamp_inplace(Tensor restrict tensor, f32 low, f32 high) {
  if (!tensor)
    return;

  Tensor res = tensor;

  auto nelement = tensor_num_elements(res);

  if (tensor->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto val                         = tensor->data->data[i + tensor->offset];
      res->data->data[i + res->offset] = clamp(val, low, high);
    }
  } else {
    u32 index[MAX_DIMS];
    do {
      auto val = tensor_get(tensor, index);
      tensor_set(res, index, clamp(val, low, high));
    } while (tensor_odometer_next(index, tensor->ndims, tensor->shape));
  }
}
