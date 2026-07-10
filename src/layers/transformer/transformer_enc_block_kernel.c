#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "transformer_enc_layer.h"
#include "var.h"

bool transformer_enc_layer_track(TransformerEncoderLayer layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->w_qkv)
    return false;

  linear_layer_track(&layer->ffn_0);
  linear_layer_track(&layer->ffn_1);
  layer_norm_track(&layer->layernorm_0);
  layer_norm_track(&layer->layernorm_1);
  layer->w_qkv = track_replace_untracked(layer->w_qkv);

  return true;
}

bool transformer_enc_layer_untrack(TransformerEncoderLayer layer) {
  TASSERT(layer && "Null layer.");

  if (!layer)
    return false;
  if (!layer->w_qkv)
    return false;

  linear_layer_untrack(&layer->ffn_0);
  linear_layer_untrack(&layer->ffn_1);
  layer_norm_untrack(&layer->layernorm_0);
  layer_norm_untrack(&layer->layernorm_1);
  untrack(layer->w_qkv);

  return true;
}

Tensor *transformer_enc_layer_kernels(TransformerEncoderLayer layer,
                                      Tensor buffer[], u32 buffer_length) {
  if (!buffer)
    return nullptr;
  if (buffer_length < N_XFORMER_KERNELS)
    return nullptr;

  buffer[0] = layer->w_qkv;
  linear_layer_kernels(&layer->ffn_0, &buffer[1], N_LINEAR_KERNELS);
  linear_layer_kernels(&layer->ffn_1, &buffer[3], N_LINEAR_KERNELS);
#define INDX5 5
#define INDX7 7
  layer_norm_kernels(&layer->layernorm_0, &buffer[INDX5], N_LAYERNORM_KERNELS);
  layer_norm_kernels(&layer->layernorm_1, &buffer[INDX7], N_LAYERNORM_KERNELS);

  return buffer;
}
