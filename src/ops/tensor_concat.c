#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include <string.h>
#include <strings.h>

Tensor tensor_concat(u32 n_tensor, Tensor *tensor) {
  u32 shape[MAX_DIMS];
  u32 ndims = tensor[0]->ndims;
  bool all_contiguous = true;
  for (u32 i = 0; i < ndims; ++i) {
    shape[i] = tensor[0]->shape[i];
  }

  for (u32 i = 1; i < n_tensor; ++i) {
    if (tensor[i]->ndims != ndims)
      return nullptr;

    for (u32 j = 0; j < ndims; ++j) {
      if (tensor[i]->shape[j] != shape[j])
        return nullptr;
    }

    all_contiguous = all_contiguous && tensor[i]->is_contiguous;
    shape[0] += tensor[i]->shape[0];
  }

  auto concat = tensor_new(ndims, shape);
  auto data = concat->data->data;

  if (all_contiguous) {
    for (u32 i = 0; i < n_tensor; ++i) {
      memcpy(data, tensor[i]->data->data,
             tensor[i]->data->nelements * sizeof(f32));
    }
  } else {
    auto index = tensor_odometer_new(ndims);
    for (u32 i = 0; i < n_tensor; ++i) {
      if (tensor[i]->is_contiguous) {
        auto nelements = tensor[i]->data->nelements;
        memcpy(data, tensor[i]->data->data, nelements * sizeof(f32));
        data += nelements;
      } else {
        tensor_odometer_reset(ndims, index);
        do {
          *data = tensor_get(tensor[i], index);
          ++data;
        } while (tensor_odometer_next(index, ndims, tensor[i]->shape));
      }
    }
    tensor_odometer_destroy(index);
  }

  return concat;
}
