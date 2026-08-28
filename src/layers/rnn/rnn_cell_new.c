#include "layers_prot.h"
#include "rnn_cell.h"
#include "tensor_memory.h"
#include <stdint.h>

RNNCell rnn_cell_new(u32 in_features, u32 batch_size, u32 hidden_features,
                     u32 out_features) {
  RNNCell layer = nullptr;
  Tensor hidden_state = tensor_new(2, (u32[]){batch_size, hidden_features});
  if (hidden_state == nullptr)
    return nullptr;

  Tensor weight =
      tensor_new(2, (u32[]){in_features + hidden_features, out_features});
  if (weight == nullptr)
    goto free_hidden_state;

  Tensor bias = tensor_new(1, (u32[]){out_features});
  if (bias == nullptr) {
    goto free_weight;
  }

  //  RNNCell layer = nullptr;
  layer = tmalloc(sizeof(*layer));
  if (layer == nullptr) {
    tensor_destroy(bias);
  free_weight:
    tensor_destroy(weight);
  free_hidden_state:
    tensor_destroy(hidden_state);
  } else {
    layer->hidden_state = hidden_state;
    layer->cell.weight = weight;
    layer->cell.bias = bias;

    tensor_xavier(layer->cell.weight, in_features, out_features);
  }

  return layer;
}
