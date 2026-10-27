#include "layers_prot.h"
#include "rnn_cell.h"
#include "tensor.h"
#include <stdio.h>

bool rnn_cell_write(RNNCell layer, FILE *binary) {
  tensor_write(layer->hidden_state, binary);
  tensor_write(layer->cell.weight, binary);
  tensor_write(layer->cell.bias, binary);
  return true;
}

RNNCell rnn_cell_read(RNNCell layer, FILE *binary) {
  layer->hidden_state = tensor_read(binary);
  layer->cell.weight  = tensor_read(binary);
  layer->cell.bias    = tensor_read(binary);
  LAYER_SET_VT(layer, rnn_cell);
  return layer;
}
