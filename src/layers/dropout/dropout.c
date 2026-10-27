#include "dropout/dropout_layer.h"
#include "layer_base.h"
#include "layers_prot.h"
#include "tensor_memory.h"
#include "var.h"
#include <stdio.h>

DropoutLayer dropout_layer_new(f32 rate) {
  DropoutLayer layer = tcalloc(1, sizeof(*layer));
  dropout_layer_init(layer, rate);
  return layer;
}
bool dropout_layer_init(DropoutLayer layer, f32 rate) {
  layer->rate        = rate;
  layer->base.vtable = &dropout_vtable;
  layer->tracking    = false;
  return true;
}
Tensor dropout_layer_forward(DropoutLayer layer, Tensor input) {
  if (!layer->tracking || var_is_tensor(input))
    return input;
  return var_dropout(input, layer->rate);
}

bool dropout_layer_destroy(DropoutLayer layer) {
  (void)layer;
  return true;
}
bool dropout_layer_deinit(DropoutLayer layer) {
  (void)layer;
  return true;
}

u32 dropout_layer_kernels(DropoutLayer layer, Tensor buffer[],
                          u32 buffer_length) {
  (void)layer;
  (void)buffer;
  (void)buffer_length;
  return 0;
}

bool dropout_layer_track(DropoutLayer layer) {
  layer->tracking = true;
  return true;
}
bool dropout_layer_untrack(DropoutLayer layer) {
  layer->tracking = false;
  return true;
}

bool dropout_layer_write(DropoutLayer layer, FILE *binary) {
  return fwrite(&layer->rate, sizeof(f32), 1, binary) != 0;
}
DropoutLayer dropout_layer_read(DropoutLayer layer, FILE *binary) {
  fread(&layer->rate, sizeof(f32), 1, binary);
  LAYER_SET_VT(layer, dropout);
  return layer;
}
