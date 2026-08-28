#include "tensor.h"

Tensor tensor_zero(const u32 ndims, const u32 *shape) {
  auto tensor = tensor_new(ndims, shape);
  return tensor;
}
