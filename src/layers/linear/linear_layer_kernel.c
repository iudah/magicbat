#include "layers_prot.h"
#include "linear_layer.h"

u32 linear_layer_kernels(LinearLayer layer, Tensor buffer[],
                         u32 buffer_length) {
  if (!buffer || buffer_length < 1)
    return 0;
  buffer[0] = layer->weight;
  if (buffer_length == 1) {
    return 1;
  }

  buffer[1] = layer->bias;

  return 2;
}
