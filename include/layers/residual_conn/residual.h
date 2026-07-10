#ifndef RESIDUAL_CONN_H
#define RESIDUAL_CONN_H

#include "layer_norm.h"
#include "tensor.h"

typedef Tensor (*ResidualStrat)(LayerNorm layernorm, Tensor input,
                                void *sublayer,
                                Tensor(sublayer_forward)(void *sublayer,
                                                         Tensor input));
Tensor pre_ln_residual(LayerNorm layernorm, Tensor input, void *sublayer,
                       Tensor(sublayer_forward)(void *sublayer, Tensor input));
Tensor post_ln_residual(LayerNorm layernorm, Tensor input, void *sublayer,
                        Tensor(sublayer_forward)(void *sublayer, Tensor input));

#endif
