#include "../../include/tensor/adt/tensor_prot.h"
#include "../../include/tensor/tensor.h"
#include "tensor_shapes_equal.h"
#include <math.h>
#include <stdint.h>

Tensor tensor_div(Tensor t, Tensor s) {
  if (!t || !s)
    return NULL;

  if (!tensor_shapes_equal(t, s))
    return NULL;

  Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
  if (res == NULL)
    return NULL;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto a = t->data[i];
    auto b = s->data[i];
    if (fabsf(b) < 1e-12) {
      tensor_destroy(res);
      return NULL;
    }
    res->data[i] = a / b;
  }

  return res;
}
