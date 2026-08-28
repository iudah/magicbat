#include "layers_prot.h"
#include "rnn_cell.h"
#include "tensor.h"
#include <stdio.h>

bool rnn_cell_serialize(RNNCell layer, FILE *binary) {
  tensor_serialize(layer->hidden_state, binary);
  tensor_serialize(layer->cell.weight, binary);
  tensor_serialize(layer->cell.bias, binary);
  return true;
}

RNNCell rnn_cell_deserialize(RNNCell layer, FILE *binary) {
  layer->hidden_state = tensor_deserialize(binary);
  layer->cell.weight = tensor_deserialize(binary);
  layer->cell.bias = tensor_deserialize(binary);
  return layer;
}
