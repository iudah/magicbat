#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "linear_layer.h"
#include "tensor.h"
#include "type_alias.h"

#define N_LAYERNORM_KERNELS N_LINEAR_KERNELS

typedef struct linear_layer *LayerNorm;

LayerNorm layer_norm_new(u32 d_model);
bool layer_norm_destroy(LayerNorm layer);
Tensor layer_norm_forward(LayerNorm layer, Tensor input);
bool layer_norm_init(LayerNorm layer, u32 d_model);
bool layer_norm_deinit(LayerNorm layer);

Tensor *layer_norm_kernels(LayerNorm layer, Tensor buffer[], u32 buffer_length);

bool layer_norm_track(LayerNorm layer);
bool layer_norm_untrack(LayerNorm layer);

#endif
