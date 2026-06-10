#include "layer_norm.h"
#include "layers_prot.h"
#include "tensor_memory.h"
#include "var.h"
#include <stdint.h>

LayerNorm layer_norm_new(u32 batch_size) {
  LayerNorm layer = nullptr;
  Tensor weight = tensor_new(1, (u32[]){batch_size});
  if (weight == nullptr)
    return nullptr;

  Tensor bias = tensor_new(1, (u32[]){batch_size});
  if (bias == nullptr) {
    goto free_weight;
  }

  //  LinearLayer layer = nullptr;
  layer = tmalloc(sizeof(*layer));
  if (layer == nullptr) {
    tensor_destroy(bias);
  free_weight:
    tensor_destroy(weight);
  } else {
    layer->weight = weight;
    layer->bias = bias; // initialzed wirh zeros

    tensor_fill(weight, 1.0F);
  }

  return layer;
}

Tensor layer_norm_forward(const LayerNorm layer, const Tensor input) {
  TASSERT(layer && input && "Null layer or input.");

  auto norm = var_layer_norm_axis(input, input->ndims - 1);
  auto weighted_sum = var_mul(norm, layer->weight);
  if (!weighted_sum)
    return nullptr;

  auto xw_b = var_add(weighted_sum, layer->bias);

  if (!var_require_grad(xw_b))
    var_destroy(weighted_sum);

  return xw_b;
}
