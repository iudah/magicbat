#include "../../../include/adt/layers/layers_prot.h"
#include "../../../include/layers/rnn/rnn_cell.h"
#include "../../../include/var/var.h"
#include "../../lifecycle/tensor_memory.h"

bool rnn_cell_destroy(RNNCell layer) {
  if (!layer)
    return false;
  if (!layer->cell.weight)
    return false;
  if (!layer->cell.bias)
    return false;

  var_destroy(layer->cell.weight);
  var_destroy(layer->cell.bias);
  var_destroy(layer->hidden_state);

  return tfree(layer);
}
