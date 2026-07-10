#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"
#include "var.h"

bool transformer_enc_layer_destroy(TransformerEncoderLayer layer) {
  if (!layer)
    return false;
  if (!layer->w_qkv)
    return false;

  layer_norm_deinit(&layer->layernorm_0);
  layer_norm_deinit(&layer->layernorm_1);
  linear_layer_deinit(&layer->ffn_0);
  linear_layer_deinit(&layer->ffn_1);
  var_destroy(layer->w_qkv);

  return tfree(layer);
}
