#include "../../../include/adt/layers/layers_prot.h"
#include "../../../include/layers/linear_layer.h"
#include "../../lifecycle/tensor_memory.h"
#include <stdint.h>

LinearLayer linear_layer_new(u32 in_features, u32 out_features) {
  LinearLayer layer = NULL;
  Tensor weight = tensor_new(2, (u32[]){in_features, out_features});
  if (weight == NULL)
    return NULL;

  Tensor bias = tensor_new(1, (u32[]){out_features});
  if (bias == NULL) {
    goto free_weight;
  }

  //  LinearLayer layer = NULL;
  layer = tmalloc(sizeof(*layer));
  if (layer == NULL) {
    tensor_destroy(bias);
  free_weight:
    tensor_destroy(weight);
  } else {
    layer->weight = weight;
    layer->bias = bias;

    // Use initializer for kernels
    // tensor_fill(weight, 0.30f);
    // tensor_fill(bias, 0.25f);
  }

  return layer;
}
