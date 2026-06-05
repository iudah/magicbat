#include "tensor.h"
#include "tensor_prot.h"
#include "tensor_shapes_equal.h"
#include <stddef.h>

bool tensor_fill(const Tensor tensor, f32 value) {
  if (tensor == nullptr)
    return false;

  for (u32 i = 0; i < tensor->data->nelements; i++) {
    tensor->data->data[i] = value;
  }

  return true;
}

bool tensor_masked_fill(const Tensor tensor, const Tensor mask, f32 value) {
  if (tensor == nullptr || mask == nullptr ||
      !tensor_shapes_equal(tensor, mask))
    return false;

#define EPS (1e-6)

  for (u32 i = 0; i < tensor->data->nelements; i++) {
    auto masked = mask->data->data[i] < EPS && mask->data->data[i] > -EPS;
    tensor->data->data[i] = masked ? 0 : value;
  }

  return true;
}
