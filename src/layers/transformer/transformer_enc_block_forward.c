#include "layers_prot.h"
#include "linear_layer.h"
#include "tensor.h"
#include "transformer_enc_layer.h"
#include "var.h"

Tensor mha_forward(void *mha_layer, Tensor pos_enc_input) {
  TransformerEncoderLayer layer = mha_layer;
  return var_multihead_attention_causal(pos_enc_input, layer->w_qkv,
                                        layer->n_head, layer->head_dim);
}
Tensor ffn_forward(void *ffn_layer, Tensor input) {
  TransformerEncoderLayer layer = ffn_layer;
  return linear_layer_forward(
      &layer->ffn_1, var_relu(linear_layer_forward(&layer->ffn_0, input)));
}
Tensor transformer_enc_layer_forward(const TransformerEncoderLayer layer,
                                     const Tensor pos_enc_input) {
  TASSERT(layer && pos_enc_input && "Null layer or input.");

  auto residual_0 = layer->residual_conn_0(&layer->layernorm_0, pos_enc_input,
                                           layer, (SublayerForward)mha_forward);
  auto residual_1 = layer->residual_conn_1(&layer->layernorm_1, residual_0,
                                           layer, (SublayerForward)ffn_forward);
  return residual_1;
}
