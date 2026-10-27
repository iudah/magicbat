#include "layers_prot.h"
#include "linear_layer.h"
#include <stdio.h>

LinearLayer linear_layer_read(LinearLayer layer, FILE *binary) {

  layer->weight = tensor_read(binary);
  layer->bias   = tensor_read(binary);
  LAYER_SET_VT(layer, linear);

  return layer;
}

bool linear_layer_write(LinearLayer layer, FILE *binary) {

  return tensor_write(layer->weight, binary) &&
         tensor_write(layer->bias, binary);
}
