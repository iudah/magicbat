#ifndef TRANSFORMER_ENC_BLOCK_H
#define TRANSFORMER_ENC_BLOCK_H

#include "residual_conn/residual.h"
#include "tensor.h"
#include "type_alias.h"

#define N_XFORMER_KERNELS 9

typedef struct transfomer_enc_layer *TransformerEncoderLayer;

TransformerEncoderLayer
transformer_enc_layer_new(u32 n_head, u32 head_dim,
                          ResidualStrat residual_strat_0,
                          ResidualStrat residual_strat_1);
bool transformer_enc_layer_destroy(TransformerEncoderLayer layer);
Tensor transformer_enc_layer_forward(TransformerEncoderLayer layer,
                                     Tensor input);

Tensor *transformer_enc_layer_kernels(TransformerEncoderLayer layer,
                                      Tensor buffer[], u32 buffer_length);

bool transformer_enc_layer_track(TransformerEncoderLayer layer);
bool transformer_enc_layer_untrack(TransformerEncoderLayer layer);

#endif
