#include "tensor.h"
#include "tensor_memory.h"
#include "tensor_prot.h"
#include "type_alias.h"
#include <stdint.h>

#define EPS (1e-6)

Tensor tensor_dropout(const Tensor tensor, u8 **mask_ptr, f32 rate) {
  TASSERT(tensor && "Null tensor");
  if (!tensor || rate <= EPS) {
    *mask_ptr = nullptr;
    if (tensor)
      atomic_fetch_add(&tensor->refcount, 1);
    return tensor;
  }

  rate       = rate >= 1 ? 1 - EPS : rate;
  Tensor res = tensor_new(tensor_ndims(tensor), tensor_shape(tensor));
  if (res == nullptr)
    return nullptr;

  tensor_copy_data(res, tensor);

#define NBITS 8
  auto nelement    = tensor_num_elements(res);
  auto mask_length = (nelement + NBITS - 1) / NBITS;
  u8  *mask        = tcalloc(mask_length, sizeof(*mask));
  auto scale       = 1 / (1 - rate);

  for (u32 i = 0; i < nelement; ++i) {
    auto drop = (urand() / (f32)UINT64_MAX < rate);
    if (drop) {
      res->data->data[i] = 0.0F;
      mask[i / NBITS] |= (u8)(1U << (i % NBITS));
    } else {
      res->data->data[i] *= scale;
    }
  }
  *mask_ptr = mask;
  return res;
}

Tensor tensor_dropout_backward(const u8 *mask, const f32 rate,
                               const Tensor grad) {
  if (!grad)
    return nullptr;

  if (rate <= EPS) {
    atomic_fetch_add(&grad->refcount, 1);
    return grad;
  }

  Tensor res = tensor_new(tensor_ndims(grad), tensor_shape(grad));
  if (res == nullptr)
    return nullptr;

  tensor_copy_data(res, grad);

  auto nelement = tensor_num_elements(res);
  auto scale    = 1 / (1 - rate);
  for (u32 i = 0; i < nelement; ++i) {
    auto drop = mask[i / NBITS] & (u8)(1U << (i % NBITS));
    if (drop) {
      res->data->data[i] = 0.0F;
    } else {
      res->data->data[i] *= scale;
    }
  }

  return res;
}
