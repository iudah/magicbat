#include "layers_prot.h"
#include "linear_layer.h"
#include <stdio.h>

LinearLayer linear_layer_deserialize(LinearLayer layer, FILE *binary) {

  layer->weight = tensor_deserialize(binary);
  layer->bias = tensor_deserialize(binary);

  return layer;
}

bool linear_layer_serialize(LinearLayer layer, FILE *binary) {

  return tensor_serialize(layer->weight, binary) &&
         tensor_serialize(layer->bias, binary);
}
