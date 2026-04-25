#include "../../include/adt/var/var_prot.h"
#include "../../include/var/var.h"
#include "../lifecycle/tensor_memory.h"
#include <stdatomic.h>
#include <stdint.h>

Tensor track(Tensor t) {
  if (!t->is_tensor_type) {
    t->requires_grad = true;
    return t;
  }

  Var v = tmalloc(sizeof(*v));

  v->base.data = t->data;
  atomic_fetch_add(&v->base.data->refcount, 1);
  v->base.ndims = t->ndims;
  v->base.requires_grad = true;
  v->base.refcount = 1;
  v->base.is_tensor_type = false;
  for (u32 i = 0; i < t->ndims; ++i) {
    v->base.shape[i] = t->shape[i];
  }
  v->grad = NULL;
  v->parent[0] = v->parent[1] = NULL;

  return (Tensor)v;
}

void untrack(Tensor v) { v->requires_grad = false; }

bool var_require_grad(Tensor v) { return v->requires_grad; }

bool var_is_tensor(Tensor v) { return v->is_tensor_type; }
