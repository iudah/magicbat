#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "data_storage.h"
#include "tensor_memory.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool tensor_destroy(Tensor t) {
  if (t == NULL)
    return false;

  if (!t->is_tensor_type)
    return false;

  if (atomic_fetch_sub(&t->refcount, 1) != 1)
    return true;

  if (t->data)
    data_storage_destroy(t->data);

  tfree(t);

  return true;
}
