#ifndef LAYER_PROT_H
#define LAYER_PROT_H

#include "../../tensor.h"

struct linear_layer {
  Tensor weight;
  Tensor bias;
};

struct rnn_cell {
  struct linear_layer cell;
};

#endif
