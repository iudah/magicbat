#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "matmul_tile.h"
#include "tensor.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"

TransformerEncoderLayer
transformer_enc_layer_new(u32 n_head, u32 head_dim,
                          ResidualStrat residual_strat_0,
                          ResidualStrat residual_strat_1) {
  auto dmodel = n_head * head_dim;
  auto dmodelff = 4 * dmodel;

  TASSERT(dmodel < BLOCK_SIZE ||
          (dmodel & (BLOCK_SIZE - 1) == 0) &&
              "Expects sequence lenght and mmodel dimensions to be a multiple "
              "of BLOCK_SIZE or less than BLOCK_SIZE.");

  if (!((dmodel < BLOCK_SIZE || ((dmodel & (BLOCK_SIZE - 1)) == 0))))
    return nullptr;

  TransformerEncoderLayer layer = tmalloc(sizeof(*layer));
  layer_norm_init(&layer->layernorm_0, dmodel);
  layer_norm_init(&layer->layernorm_1, dmodel);
  linear_layer_init(&layer->ffn_0, dmodel, dmodelff);
  tensor_kaiming(layer->ffn_0.weight, dmodel, dmodelff);
  linear_layer_init(&layer->ffn_1, dmodelff, dmodel);

  layer->residual_conn_0 = residual_strat_0;
  layer->residual_conn_1 = residual_strat_1;
  layer->w_qkv = tensor_new(2, (u32[MAX_DIMS]){dmodel, 3 * dmodel});
  layer->head_dim = head_dim;
  layer->n_head = n_head;

  tensor_xavier(layer->w_qkv, dmodel, dmodel);

  return layer;
}
