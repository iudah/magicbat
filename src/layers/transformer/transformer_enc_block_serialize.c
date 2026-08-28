#include "layers_prot.h"
#include "tensor.h"
#include "transformer_enc_layer.h"
#include <stdio.h>

TransformerEncoderLayer
transformer_enc_layer_deserialize(TransformerEncoderLayer layer, FILE *binary) {

  layer_norm_deserialize(&layer->layernorm_0, binary);
  layer_norm_deserialize(&layer->layernorm_1, binary);
  linear_layer_deserialize(&layer->ffn_0, binary);
  linear_layer_deserialize(&layer->ffn_1, binary);

  u8 pre_ln;
  fread(&pre_ln, sizeof(pre_ln), 1, binary);
  layer->residual_conn_0 =
      (pre_ln >> 4) == 0 ? post_ln_residual : pre_ln_residual;
  layer->residual_conn_1 =
      (pre_ln & 0xf) == 0 ? post_ln_residual : pre_ln_residual;
  layer->w_qkv = tensor_deserialize(binary);
  fread(&layer->head_dim, sizeof(layer->head_dim), 1, binary);
  fread(&layer->n_head, sizeof(layer->n_head), 1, binary);

  return layer;
}

bool transformer_enc_layer_serialize(TransformerEncoderLayer layer,
                                     FILE *binary) {
  layer_norm_serialize(&layer->layernorm_0, binary);
  layer_norm_serialize(&layer->layernorm_1, binary);
  linear_layer_serialize(&layer->ffn_0, binary);
  linear_layer_serialize(&layer->ffn_1, binary);

  u8 pre_ln = (u8)(layer->residual_conn_0 == pre_ln_residual) << 4 |
              (u8)(layer->residual_conn_1 == pre_ln_residual);
  fwrite(&pre_ln, sizeof(pre_ln), 1, binary);
  tensor_serialize(layer->w_qkv, binary);
  fwrite(&layer->head_dim, sizeof(layer->head_dim), 1, binary);
  fwrite(&layer->n_head, sizeof(layer->n_head), 1, binary);

  return true;
}
