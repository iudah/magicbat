
#include "../../../include/adt/layers/layers_prot.h"
#include "../../../include/layers/rnn/rnn_cell.h"
#include "../../../include/var/var.h"

bool rnn_cell_track(RNNCell layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->cell.weight)
    return false;
  if (!layer->cell.bias)
    return false;

  auto weight = track(layer->cell.weight);
  if (var_is_tensor(layer->cell.weight)) {
    tensor_destroy(layer->cell.weight);
    layer->cell.weight = weight;
  }

  auto bias = track(layer->cell.bias);
  if (var_is_tensor(layer->cell.bias)) {
    tensor_destroy(layer->cell.bias);
    layer->cell.bias = bias;
  }

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
