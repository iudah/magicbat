#include "../../include/adt/layers/layers_prot.h"
#include "../../include/layers/linear_layer.h"
#include "../lifecycle/tensor_memory.h"

bool linear_layer_destroy(LinearLayer layer) {
  if (!layer)
    return false;
  if (!layer->weight)
    return false;
  if (!layer->bias)
    return false;

  tensor_destroy(layer->weight);
  tensor_destroy(layer->bias);

  return tfree(layer);
}
