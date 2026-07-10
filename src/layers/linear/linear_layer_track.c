#include "layers_prot.h"
#include "linear_layer.h"
#include "var.h"

bool linear_layer_track(LinearLayer layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->weight)
    return false;
  if (!layer->bias)
    return false;

  auto weight = track(layer->weight);
  if (var_is_tensor(layer->weight)) {
    tensor_destroy(layer->weight);
    layer->weight = weight;
  }

  auto bias = track(layer->bias);
  if (var_is_tensor(layer->bias)) {
    tensor_destroy(layer->bias);
    layer->bias = bias;
  }

  return true;
}

bool linear_layer_untrack(LinearLayer layer) {
  TASSERT(layer && "Null layer.");

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
