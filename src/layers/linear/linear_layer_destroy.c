#include "layers_prot.h"
#include "linear_layer.h"
#include "tensor_memory.h"
#include "var.h"

bool linear_layer_deinit(LinearLayer layer) {
  if (!layer)
    return false;
  if (!layer->weight)
    return false;
  if (!layer->bias)
    return false;

  var_destroy(layer->weight);
  var_destroy(layer->bias);

  return true;
}

bool linear_layer_destroy(LinearLayer layer) {
  return linear_layer_deinit(layer) && tfree(layer);
}
