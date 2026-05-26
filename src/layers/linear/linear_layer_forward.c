#include "../../../include/adt/layers/layers_prot.h"
#include "../../../include/layers/linear_layer.h"
#include "../../../include/var/var.h"

Tensor linear_layer_forward(const LinearLayer layer, const Tensor in) {
  TASSERT(layer && in && "Null layer or input.");

  // To Do: Use tensordot
  auto xw = var_matmul(in, layer->weight);
  if (!xw)
    return NULL;

  auto xw_b = var_add((Tensor)xw, layer->bias);

  if (!var_require_grad(xw_b))
    var_destroy(xw);

  return (Tensor)xw_b;
}
