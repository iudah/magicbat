#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "tensor.h"
#include "type_alias.h"

#define N_LINEAR_KERNELS 2

typedef struct linear_layer *LinearLayer;


LinearLayer linear_layer_new(u32 in_feature, u32 out_feature);
bool linear_layer_destroy(LinearLayer layer);
Tensor linear_layer_forward(LinearLayer layer, Tensor input);
bool linear_layer_init(LinearLayer layer, u32 in_features, u32 out_features);
bool linear_layer_deinit(LinearLayer layer);

Tensor *linear_layer_kernels(LinearLayer layer, Tensor buffer[],
                             u32 buffer_length);

bool linear_layer_track(LinearLayer layer);
bool linear_layer_untrack(LinearLayer layer);

#endif
