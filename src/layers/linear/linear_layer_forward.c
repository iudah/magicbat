#include "layers_prot.h"
#include "linear_layer.h"
#include "var.h"

Tensor linear_layer_forward(const LinearLayer layer, const Tensor input) {
  TASSERT(layer && in && "Null layer or input.");

  auto weighted_sum = var_bmm(input, layer->weight);
  if (!weighted_sum)
    return nullptr;

  auto xw_b = var_add(weighted_sum, layer->bias);

  if (!var_require_grad(xw_b))
    var_destroy(weighted_sum);

  return xw_b;
}
