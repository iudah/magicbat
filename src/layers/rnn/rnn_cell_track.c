
#include "layers_prot.h"
#include "rnn_cell.h"
#include "var.h"

bool rnn_cell_track(RNNCell layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->cell.weight)
    return false;
  if (!layer->cell.bias)
    return false;

  layer->cell.weight = to_var(layer->cell.weight);

  layer->cell.bias = to_var(layer->cell.bias);

  layer->hidden_state = to_var(layer->hidden_state);

  return true;
}

bool rnn_cell_untrack(RNNCell layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->cell.weight)
    return false;
  if (!layer->cell.bias)
    return false;

  untrack(layer->cell.weight);
  untrack(layer->cell.bias);
  untrack(layer->hidden_state);

  return true;
}
