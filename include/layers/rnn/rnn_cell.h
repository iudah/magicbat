#ifndef RNN_CELL_H
#define RNN_CELL_H

#include "tensor.h"
#include "type_alias.h"

typedef struct rnn_cell *RNNCell;

RNNCell rnn_cell_new(u32 in_features, u32 batch_size, u32 hidden_features,
                     u32 out_features);
bool rnn_cell_destroy(RNNCell layer);
Tensor rnn_cell_forward(RNNCell layer, Tensor input);

Tensor rnn_cell_weight(RNNCell layer);
Tensor rnn_cell_bias(RNNCell layer);

bool rnn_cell_track(RNNCell layer);
bool rnn_cell_untrack(RNNCell layer);

#endif
