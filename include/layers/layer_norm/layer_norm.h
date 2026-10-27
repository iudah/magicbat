#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "layer_base.h"
#include "layer_bin.h"
#include "linear_layer.h"
#include "tensor.h"
#include "type_alias.h"

#define N_LAYERNORM_KERNELS N_LINEAR_KERNELS

typedef struct linear_layer *LayerNorm;

LayerNorm layer_norm_new(u32 d_model);
bool      layer_norm_destroy(LayerNorm layer);
Tensor    layer_norm_forward(LayerNorm layer, Tensor input);
bool      layer_norm_init(LayerNorm layer, u32 d_model);
bool      layer_norm_deinit(LayerNorm layer);

u32 layer_norm_kernels(LayerNorm layer, Tensor buffer[], u32 buffer_length);

bool layer_norm_track(LayerNorm layer);
bool layer_norm_untrack(LayerNorm layer);

bool      layer_norm_write(LayerNorm layer, FILE *binary);
LayerNorm layer_norm_read(LayerNorm layer, FILE *binary);

DEFINE_VTABLE(layer_norm, layer_norm, LayerNorm, "LayerNorm",
              N_LAYERNORM_KERNELS, LAYERNORM_SERIALIZER)

#endif
