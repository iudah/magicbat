#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/data_storage.h"
#include "../lifecycle/tensor_memory.h"
#include <stdint.h>

Tensor tensor_scalar(f32 value) {
  Tensor t = tmalloc(sizeof(*t));
  if (t == nullptr)
    return nullptr;

  t->ndims = 1;

  t->data = data_storage_new(1);

  *t->data->data = value;
  t->shape[0] = 1;

  t->refcount = 1;
  t->requires_grad = false;
  t->is_tensor_type = true;

  return t;
}
