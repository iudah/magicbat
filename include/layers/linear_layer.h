#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "tensor.h"
#include "type_alias.h"

typedef struct linear_layer *LinearLayer;

LinearLayer linear_layer_new(u32 in_feature, u32 out_feature);
bool linear_layer_destroy(LinearLayer layer);
Tensor linear_layer_forward(const LinearLayer layer, const Tensor in);

Tensor linear_layer_weight(const LinearLayer layer);
Tensor linear_layer_bias(const LinearLayer layer);

bool linear_layer_track(LinearLayer layer);
bool linear_layer_untrack(LinearLayer layer);

#endif
