#include "layers_prot.h"
#include "linear_layer.h"
#include "tensor.h"
#include "tensor_memory.h"
#include <stdint.h>

bool linear_layer_init(LinearLayer layer, u32 in_features, u32 out_features) {
  bool retval = true;
  Tensor weight = tensor_new(2, (u32[]){in_features, out_features});
  if (weight == nullptr)
    return false;

  Tensor bias = tensor_new(1, (u32[]){out_features});
  if (bias == nullptr) {
    goto free_weight;
  }

  if (layer == nullptr) {
    tensor_destroy(bias);
  free_weight:
    tensor_destroy(weight);

    retval = false;
  } else {
    layer->weight = weight;
    layer->bias = bias;

    tensor_xavier(weight, in_features, out_features);
  }

  return retval;
}

LinearLayer linear_layer_new(u32 in_features, u32 out_features) {
  LinearLayer layer = tmalloc(sizeof(*layer));

  if (!linear_layer_init(layer, in_features, out_features))
    return nullptr;

  return layer;
}
