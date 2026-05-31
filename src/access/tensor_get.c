#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_ndim_flat_index.h"
#include <math.h>

f32 tensor_get(const Tensor tensor, const u32 *index) {
  TASSERT(tensor && "Null tensor");

  auto flat = tensor_ndim_flat_index(tensor, index);

  if (flat >= tensor->data->nelements)
    return NAN;

  return tensor->data->data[flat];
}
