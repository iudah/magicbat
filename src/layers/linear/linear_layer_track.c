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

  layer->weight = to_var(layer->weight);

  layer->bias = to_var(layer->bias);

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
