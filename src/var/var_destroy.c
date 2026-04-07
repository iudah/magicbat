#include "../../include/adt/var/var_prot.h"
#include "../../include/var/var.h"
#include "../lifecycle/data_storage.h"
#include "../lifecycle/tensor_memory.h"

bool var_destroy(Var v) {
  if (!v)
    return false;

  if (v->base.is_tensor_type)
    return false;

  if (atomic_fetch_sub(&v->base.refcount, 1) != 1)
    return true;

  data_storage_destroy(v->base.data);

  if (v->parent[0])
    var_destroy(v->parent[0]);

  if (v->parent[1])
    var_destroy(v->parent[1]);

  if (v->ctx) {
    if (v->op.destroy_ctx)
      v->op.destroy_ctx(v->ctx);
    else
      tfree(v->ctx);
  }

  tfree(v);
  return true;
}
