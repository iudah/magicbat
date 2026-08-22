#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "tensor_memory.h"
#include "var.h"
#include <stdint.h>

bool layer_norm_init(LayerNorm layer, u32 d_model) {
  auto retval = false;

  Tensor weight = tensor_new(1, (u32[]){d_model});
  if (weight == nullptr)
    return retval;

  Tensor bias = tensor_new(1, (u32[]){d_model});
  if (bias == nullptr) {
    goto free_weight;
  }

  if (layer == nullptr) {
    tensor_destroy(bias);
  free_weight:
    tensor_destroy(weight);
  } else {
    layer->weight = weight;
    layer->bias = bias; // initialzed wirh zeros

    tensor_fill(weight, 1.0F);
    retval = true;
  }

  return retval;
}

LayerNorm layer_norm_new(u32 d_model) {
  LayerNorm layer = tmalloc(sizeof(*layer));
  if (!layer_norm_init(layer, d_model))
    return nullptr;
  return layer;
}

Tensor layer_norm_forward(const LayerNorm layer, const Tensor input) {
  TASSERT(layer && input && "Null layer or input.");

  auto norm = var_layer_norm_axis(input, input->ndims - 1);
  if (!norm)
    return nullptr;

  auto weighted_sum = var_mul(norm, layer->weight);
  if (!weighted_sum)
    return nullptr;

  auto xw_b = var_add(weighted_sum, layer->bias);

  if (!var_require_grad(xw_b))
    var_destroy(weighted_sum);

  return xw_b;
}

bool layer_norm_deinit(LayerNorm layer) { return linear_layer_deinit(layer); }

bool layer_norm_destroy(LayerNorm layer) {
  return layer_norm_deinit(layer) && tfree(layer);
}

Tensor *layer_norm_kernels(LayerNorm layer, Tensor buffer[],
                           u32 buffer_length) {
  return linear_layer_kernels(layer, buffer, buffer_length);
}

bool layer_norm_track(LayerNorm layer) { return linear_layer_track(layer); }
bool layer_norm_untrack(LayerNorm layer) { return linear_layer_untrack(layer); }
