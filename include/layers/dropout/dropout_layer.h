#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer_base.h"
#include "layer_bin.h"
#include "tensor.h"
#include "type_alias.h"

typedef struct dropout_layer *DropoutLayer;

DropoutLayer dropout_layer_new(f32 rate);
bool         dropout_layer_destroy(DropoutLayer layer);
Tensor       dropout_layer_forward(DropoutLayer layer, Tensor input);
bool         dropout_layer_init(DropoutLayer layer, f32 rate);
bool         dropout_layer_deinit(DropoutLayer layer);

u32 dropout_layer_kernels(DropoutLayer layer, Tensor buffer[],
                          u32 buffer_length);

bool dropout_layer_track(DropoutLayer layer);
bool dropout_layer_untrack(DropoutLayer layer);

bool         dropout_layer_write(DropoutLayer layer, FILE *binary);
DropoutLayer dropout_layer_read(DropoutLayer layer, FILE *binary);

DEFINE_VTABLE(dropout, dropout_layer, DropoutLayer, "DropoutLayer", 0,
              DROPOUT_SERIALIZER)

#endif
