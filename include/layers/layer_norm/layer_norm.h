#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"
#include "type_alias.h"

typedef struct linear_layer *LayerNorm;

LayerNorm layer_norm_new(u32 batch_size);
bool layer_norm_destroy(LayerNorm layer);
Tensor layer_norm_forward(LayerNorm layer, Tensor input);

Tensor layer_norm_weight(LayerNorm layer);
Tensor layer_norm_bias(LayerNorm layer);

bool layer_norm_track(LayerNorm layer);
bool layer_norm_untrack(LayerNorm layer);

#endif
