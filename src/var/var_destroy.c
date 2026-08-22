#include "../../include/adt/var/var_prot.h"
#include "../../include/var/var.h"
#include "../lifecycle/data_storage.h"
#include "../lifecycle/tensor_memory.h"
#include "tensor.h"
#include "type_alias.h"
#include <stdint.h>

static void var_destroy_grad(Var variable) {
  if (variable->grad && variable->base.is_tensor_type) {
    tensor_destroy(variable->grad);
  }
  if (variable->grad && !variable->base.is_tensor_type) {
    var_destroy(variable->grad);
  }
}

bool var_destroy(Tensor var) {
  Var variable = (Var)var;

  if (!variable)
    return false;

  if (variable->base.is_tensor_type)
    return tensor_destroy((Tensor)variable);

#define capacity (32)
  u32 cap = capacity;
  u32 idx = 0;
  Var *vars = tmalloc(cap * sizeof(*vars));
  u32 last = 0;
  vars[last] = variable;
  last = 1;

  while (idx != last) {
    variable = vars[idx++];

    if (atomic_fetch_sub(&variable->base.refcount, 1) != 1)
      continue;

    data_storage_destroy(variable->base.data);
    var_destroy_grad(variable);

    if (variable->base.is_tensor_type) {
      tensor_destroy((Tensor)variable);
      continue;
    }

    if (variable->parent[0]) {
      vars[last++] = (Var)(variable->parent[0]);
      if (last == cap) {
        cap <<= 1;
        vars = trealloc(vars, cap * sizeof(*vars));
      }
    }

    if (variable->parent[1]) {
      vars[last++] = (Var)(variable->parent[1]);
      if (last == cap) {
        cap <<= 1;
        vars = trealloc(vars, cap * sizeof(*vars));
      }
    }

    if (variable->ctx) {
      if (variable->op.destroy_ctx)
        variable->op.destroy_ctx(variable->ctx);
      else
        tfree(variable->ctx);
    }

    tfree(variable);
  }
  tfree(vars);
  return true;
}
