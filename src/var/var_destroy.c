#include "../../include/adt/var/var_prot.h"
#include "../../include/var/var.h"
#include "../lifecycle/data_storage.h"
#include "../lifecycle/tensor_memory.h"
#include <stdint.h>

bool var_destroy(Var v) {
  if (!v)
    return false;

  if (v->base.is_tensor_type)
    return tensor_destroy((Tensor)v);

  u32 cap = 32;
  u32 idx = 0;
  Var *vars = tmalloc(cap * sizeof(*vars));
  u32 last = 0;
  vars[last] = v;
  last = 1;

  while (idx != last) {
    v = vars[idx++];

    if (atomic_fetch_sub(&v->base.refcount, 1) != 1)
      continue;

    data_storage_destroy(v->base.data);

    if (v->base.is_tensor_type) {
      tensor_destroy((Tensor)v);
      continue;
    }

    if (v->parent[0]) {
      vars[last++] = (v->parent[0]);
      if (last == cap) {
        cap <<= 1;
        vars = trealloc(vars, cap * sizeof(*vars));
      }
    }

    if (v->parent[1]) {
      vars[last++] = (v->parent[1]);
      if (last == cap) {
        cap <<= 1;
        vars = trealloc(vars, cap * sizeof(*vars));
      }
    }

    if (v->ctx) {
      if (v->op.destroy_ctx)
        v->op.destroy_ctx(v->ctx);
      else
        tfree(v->ctx);
    }

    tfree(v);
  }
  return true;
}
