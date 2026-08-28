#include "tensor.h"
#include "tensor_binary_op.h"
#include "tensor_prot.h"
#include "tensor_shapes_equal.h"
#include "type_alias.h"
#include <stddef.h>

f32 fill_fn(f32 __attribute__((unused)) float_a, f32 value) { return value; }

bool tensor_fill(const Tensor tensor, f32 value) {
  if (tensor == nullptr)
    return false;

  if (!tensor->is_contiguous) {
    for (u32 i = 0; i < tensor->data->nelements; i++) {
      tensor->data->data[tensor->offset + i] = value;
    }
  } else
    return tensor_binary_op_scalar_inplace(tensor, value, fill_fn);

  return true;
}

#define EPS (1e-6)

f32 masked_fill_fn(f32 __attribute__((unused)) float_a, f32 mask, f32 value) {
  auto masked = mask > EPS || mask < -EPS;
  return masked ? 0 : value;
}

bool tensor_masked_fill(const Tensor tensor, const Tensor mask, f32 value) {
  if (tensor == nullptr || mask == nullptr ||
      !tensor_shapes_equal(tensor, mask))
    return false;

  if (tensor->is_contiguous && mask->is_contiguous) {
    for (u32 i = 0; i < tensor->data->nelements; i++) {
      auto masked = mask->data->data[mask->offset + i] < EPS &&
                    mask->data->data[mask->offset + i] > -EPS;
      tensor->data->data[tensor->offset + i] = masked ? 0 : value;
    }

  } else
    return tensor_binary_op_inplace(tensor, mask, value, masked_fill_fn);

  return true;
}
