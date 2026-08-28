#include "var_prot.h"
#include "tensor.h"
#include "tensor_memory.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

bool var_zero_grad(Tensor top) {

  Var variable = (Var)top;

  if (!top || top->is_tensor_type || !top->requires_grad)
    return false;

#define capacity (32)
  u32 cap = capacity;
  Var *list = tmalloc(cap * sizeof(*list));
  u32 count = 0;
  mem top_grad = variable->grad;
  variable->grad = nullptr;

  list[count++] = variable;
  for (u32 i = 0; i < count; ++i) {
    Var var = list[i];
    if (var->grad && !var->parent[0] && !var->parent[1]) {
      tensor_fill(var->grad, 0);
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
    if (var->parent[0] && !var->parent[0]->base.is_tensor_type &&
        var->parent[0]->base.requires_grad)
      list[count++] = var->parent[0];
    if (var->parent[1] && !var->parent[1]->base.is_tensor_type &&
        var->parent[1]->base.requires_grad)
      list[count++] = var->parent[1];
  }

  variable->grad = top_grad;

  tfree(list);
  return true;
}
