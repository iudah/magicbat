#include "../../../include/adt/layers/layers_prot.h"
#include "../../../include/layers/rnn/rnn_cell.h"

Tensor rnn_cell_weight(const RNNCell layer) { return layer->cell.weight; }
Tensor rnn_cell_bias(const RNNCell layer) { return layer->cell.bias; }
