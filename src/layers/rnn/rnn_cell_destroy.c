#include "layers_prot.h"
#include "rnn_cell.h"
#include "tensor_memory.h"
#include "var.h"

bool rnn_cell_deinit(RNNCell layer) {
  if (!layer)
    return false;
  if (!layer->cell.weight)
    return false;
  if (!layer->cell.bias)
    return false;
  var_destroy(layer->cell.weight);
  var_destroy(layer->cell.bias);
  var_destroy(layer->hidden_state);
  return true;
}

bool rnn_cell_destroy(RNNCell layer) {
  return rnn_cell_deinit(layer) && tfree(layer);
}
