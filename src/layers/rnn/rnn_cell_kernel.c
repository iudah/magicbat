#include "layers_prot.h"
#include "rnn_cell.h"

Tensor rnn_cell_weight(const RNNCell layer) { return layer->cell.weight; }
Tensor rnn_cell_bias(const RNNCell layer) { return layer->cell.bias; }
