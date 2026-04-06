#include "../../include/adt/layers/layers_prot.h"
#include "../../include/layers/linear_layer.h"

Tensor linear_layer_weight(const LinearLayer layer) { return layer->weight; }
Tensor linear_layer_bias(const LinearLayer layer) { return layer->bias; }
