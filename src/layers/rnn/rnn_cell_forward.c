#include "layers_prot.h"
#include "rnn_cell.h"
#include "var.h"

Tensor rnn_cell_forward(const RNNCell layer, const Tensor input) {
  TASSERT(layer && input && "Null layer or input.");

  auto fused = var_concat(2, (Tensor[]){input, layer->hidden_state}, 1);

  if (!fused) {
    return nullptr;
  }

  auto weighted_sum = var_matmul(fused, layer->cell.weight);
  if (!weighted_sum)
    return nullptr;

  auto xw_b = var_add(weighted_sum, layer->cell.bias);

  if (!var_require_grad(xw_b))
    var_destroy(weighted_sum);

  auto hidden_state = layer->hidden_state;
  layer->hidden_state = var_tanh(xw_b);
  var_destroy(hidden_state);

  return layer->hidden_state;
}
