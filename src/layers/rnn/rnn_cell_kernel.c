#include "layers_prot.h"
#include "rnn_cell.h"

Tensor rnn_cell_weight(const RNNCell layer) { return layer->cell.weight; }
Tensor rnn_cell_bias(const RNNCell layer) { return layer->cell.bias; }
u32    rnn_cell_kernels(RNNCell layer, Tensor buffer[], u32 buffer_length) {
  if (!buffer || buffer_length < 1)
    return 0;
  buffer[0] = layer->cell.weight;
  if (buffer_length == 1) {
    return 1;
  }

  buffer[1] = layer->cell.bias;

  return 2;
}
