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
  layer->w_qkv = to_var(layer->w_qkv);

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

u32 transformer_enc_layer_kernels(TransformerEncoderLayer layer,
                                  Tensor buffer[], u32 buffer_length) {
  if (!buffer || buffer_length < 1)
    return 0;

  buffer[0] = layer->w_qkv;
  if (buffer_length == 1)
    return 1;

  auto len = linear_layer_kernels(&layer->ffn_0, buffer + 1, buffer_length - 1);
  len += linear_layer_kernels(&layer->ffn_1, buffer + 1 + len,
                              buffer_length - 1 - len);
  len += layer_norm_kernels(&layer->layernorm_0, buffer + 1 + len,
                            buffer_length - 1 - len);
  len += layer_norm_kernels(&layer->layernorm_1, buffer + 1 + len,
                            buffer_length - 1 - len);

  return N_XFORMER_KERNELS;
}
