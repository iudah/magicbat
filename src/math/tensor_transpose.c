#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_transpose(const Tensor tensor) {
  TASSERT(t && "Null tensor.");

  if (!tensor)
    return nullptr;
  if (tensor->ndims != 2)
    return nullptr;

  /*
 u32 row = tensor->shape[0];
   u32 col = tensor->shape[1];

   // guarantees zeros
   Tensor res = tensor_new(2, (u32[]){col, row});
   if (res == nullptr)
     return nullptr;

   for (u32 i = 0; i < row; ++i) {
     float *t_data = &tensor->data->data[i * col];
     for (u32 j = 0; j < col; ++j) {
       float t_val = t_data[j];
       res->data->data[j * row + i] = t_val;
     }
   }

   return res;
 */

  return tensor_transpose_dims(tensor, 1,
                               (TensorDimSwap[]){{.src = 0, .dest = 1}});
}

Tensor tensor_transpose_dims(const Tensor tensor, u32 nswap,
                             TensorDimSwap swaps[]) {
  auto view = tensor_view(tensor);
  for (u32 i = 0; i < nswap; ++i) {
    view->shape[swaps[i].dest] = tensor->shape[swaps[i].src];
    view->shape[swaps[i].src] = tensor->shape[swaps[i].dest];

    view->stride[swaps[i].dest] = tensor->stride[swaps[i].src];
    view->stride[swaps[i].src] = tensor->stride[swaps[i].dest];
  }

  return view;
}
