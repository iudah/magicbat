#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>

static inline f32 sqr(f32 num) { return num * num; }

f32 tensor_total_norm_sqr(Tensor tensor) {
  TASSERT(tensor && "Null tensor.");

  if (!tensor)
    return -INFINITY;

  float sum       = 0;
  u32   nelements = tensor_num_elements(tensor);
  if (tensor->is_contiguous) {
    for (u32 i = 0; i < nelements; ++i) {
      sum += sqr(tensor->data->data[i + tensor->offset]);
    }
  } else {
    u32 index[MAX_DIMS];
    do {
      sum += sqr(tensor_get(tensor, index));
    } while (tensor_odometer_next(index, tensor->ndims, tensor->shape));
  }

  return sum;
}
