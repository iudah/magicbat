#ifndef RNN_CELL_H
#define RNN_CELL_H

#include "layer_bin.h"
#include "layers_prot.h"
#include "tensor.h"
#include "type_alias.h"

typedef struct rnn_cell *RNNCell;

RNNCell rnn_cell_new(u32 in_features, u32 batch_size, u32 hidden_features,
                     u32 out_features);
bool    rnn_cell_destroy(RNNCell layer);
Tensor  rnn_cell_forward(RNNCell layer, Tensor input);

Tensor rnn_cell_weight(RNNCell layer);
Tensor rnn_cell_bias(RNNCell layer);
u32    rnn_cell_kernels(RNNCell layer, Tensor buffer[], u32 buffer_length);

bool rnn_cell_track(RNNCell layer);
bool rnn_cell_untrack(RNNCell layer);

bool    rnn_cell_write(RNNCell layer, FILE *binary);
RNNCell rnn_cell_read(RNNCell layer, FILE *binary);

DEFINE_VTABLE(rnn_cell, rnn_cell, RNNCell, "RNNCell", 2, RNNCELL_SERIALIZER)
#endif
