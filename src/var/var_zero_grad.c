#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

bool var_zero_grad(Tensor top_) {

  Var top = (Var)top_;

  Tensor a = (Tensor)top;
  if (!a || a->is_tensor_type || !a->requires_grad)
    return false;

  u32 cap = 32;
  Var *list = tmalloc(cap * sizeof(*list));
  u32 count = 0;
  mem top_grad = top->grad;
  top->grad = nullptr;

  list[count++] = top;
  for (u32 i = 0; i < count; ++i) {
    Var v = list[i];
    if (v->grad) {
      tensor_fill(v->grad, 0);
    }

    if (cap - count < 2) {
      u32 new_cap = cap * 2;
      mem tmp = trealloc(list, new_cap * sizeof(*list));
      if (!tmp) {
        tfree(list);
        return false;
      }
      list = tmp;
      cap = new_cap;
    }
    if (v->parent[0] && !v->parent[0]->base.is_tensor_type &&
        v->parent[0]->base.requires_grad)
      list[count++] = v->parent[0];
    if (v->parent[1] && !v->parent[1]->base.is_tensor_type &&
        v->parent[1]->base.requires_grad)
      list[count++] = v->parent[1];
  }

  top->grad = top_grad;

  tfree(list);
  return true;
}
