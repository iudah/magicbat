#include "../../include/adt/var/var_prot.h"
#include "../../include/var/var.h"
#include "../lifecycle/tensor_memory.h"
#include <stdatomic.h>
#include <stdint.h>

Tensor track(Tensor tensor) {
  if (!tensor->is_tensor_type) {
    tensor->requires_grad = true;
    return tensor;
  }

  Var var = tmalloc(sizeof(*var));

  var->base.data = tensor->data;
  atomic_fetch_add(&var->base.data->refcount, 1);
  var->base.ndims = tensor->ndims;
  var->base.requires_grad = true;
  var->base.refcount = 1;
  var->base.is_tensor_type = false;
  var->base.is_contiguous = tensor->is_contiguous;
  var->base.offset = tensor->offset;
  for (u32 i = 0; i < tensor->ndims; ++i) {
    var->base.shape[i] = tensor->shape[i];
    var->base.stride[i] = tensor->stride[i];
  }
  var->grad = nullptr;
  var->parent[0] = var->parent[1] = nullptr;

  return (Tensor)var;
}

void untrack(Tensor var) { var->requires_grad = false; }

bool var_require_grad(Tensor var) { return var->requires_grad; }

bool var_is_tensor(Tensor var) { return var->is_tensor_type; }
