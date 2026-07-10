#include "layer_norm.h"
#include "var.h"

Tensor pre_ln_residual(LayerNorm layernorm, Tensor input, void *sublayer,
                       Tensor(sublayer_forward)(void *sublayer, Tensor input)) {
  return var_add(
      input, layer_norm_forward(layernorm, sublayer_forward(sublayer, input)));
}

Tensor post_ln_residual(LayerNorm layernorm, Tensor input, void *sublayer,
                        Tensor(sublayer_forward)(void *sublayer,
                                                 Tensor input)) {
  return layer_norm_forward(layernorm,
                            var_add(input, sublayer_forward(sublayer, input)));
}
