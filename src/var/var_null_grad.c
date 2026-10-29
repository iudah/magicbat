#include "tensor.h"
#include "var_prot.h"
#include <stdint.h>

void var_null_grad(Tensor tensor) {
  if (!tensor || tensor->is_tensor_type)
    return;

  Var var = (Var)tensor;
  if (!var->grad)
    return;

#if defined(ARENA_USE_MALLOC) || defined(USE_MGRIND)
  tensor_fill(var->grad, 0.0F);
#else
  var->grad = nullptr;
#endif
}

u32 var_null_grads(Tensor *params, u32 n) {
  if (!params || n == 0)
    return 0;

  u32 discarded = 0;
  for (u32 i = 0; i < n; ++i) {
    Tensor param = params[i];
    if (!param || param->is_tensor_type)
      continue;

    Var var = (Var)param;
    if (!var->grad)
      continue;

    var_null_grad(param);
    ++discarded;
  }
  return discarded;
}
