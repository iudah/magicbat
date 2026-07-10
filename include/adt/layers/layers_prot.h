#ifndef LAYER_PROT_H
#define LAYER_PROT_H

#include "tensor.h"

struct linear_layer {
  Tensor weight;
  Tensor bias;
};

struct layer_norm {
  struct linear_layer linear;
};

struct rnn_cell {
  struct linear_layer cell;
  // Untrack hidden_state  before  next sequence.
  Tensor hidden_state;
};

typedef Tensor (*SublayerForward)(void *sublayer, Tensor input);

struct transfomer_enc_layer {
  Tensor (*residual_conn_0)(struct linear_layer *layernorm, Tensor input,
                            void *sublayer, SublayerForward sublayer_forward);
  Tensor (*residual_conn_1)(struct linear_layer *layernorm, Tensor input,
                            void *sublayer, SublayerForward sublayer_forward);
  struct linear_layer ffn_0;
  struct linear_layer ffn_1;
  struct linear_layer layernorm_0;
  struct linear_layer layernorm_1;
  Tensor w_qkv;
  u32 n_head;
  u32 head_dim;
};

#endif
