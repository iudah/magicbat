
#include "../../include/adt/layers/layers_prot.h"
#include "../../include/layers/linear_layer.h"
#include "../../include/var/var.h"

bool linear_layer_track(LinearLayer layer) {
  if (!layer)
    return false;
  if (!layer->weight)
    return false;
  if (!layer->bias)
    return false;

  auto w = track(layer->weight);
  if (var_is_tensor(layer->weight)) {
    tensor_destroy(layer->weight);
    layer->weight = (Tensor)w;
  }

  auto b = track(layer->bias);
  if (var_is_tensor(layer->bias)) {
    tensor_destroy(layer->bias);
    layer->bias = (Tensor)b;
  }

  return true;
}

bool linear_layer_untrack(LinearLayer layer);
bool linear_layer_untrack(LinearLayer layer) {
  if (!layer)
    return false;
  if (!layer->weight)
    return false;
  if (!layer->bias)
    return false;

  untrack(layer->weight);

  untrack(layer->bias);

  return true;
}
