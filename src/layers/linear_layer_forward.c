#include "../../include/adt/layers/layers_prot.h"
#include "../../include/layers/linear_layer.h"

Tensor linear_layer_forward(const LinearLayer layer, const Tensor in) {
  // To Do: Use tensordot
  auto xw = tensor_matmul(in, layer->weight);
  if (!xw)
    return NULL;

  auto xw_b = tensor_add(xw, layer->bias);
  tensor_destroy(xw);

  return xw_b;
}
