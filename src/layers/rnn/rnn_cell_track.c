
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

  layer->cell.weight = track_replace_untracked(layer->cell.weight);

  layer->cell.bias = track_replace_untracked(layer->cell.bias);

  layer->hidden_state = track_replace_untracked(layer->hidden_state);

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
