#include "layers_prot.h"
#include "linear_layer.h"


Tensor *linear_layer_kernels(LinearLayer layer, Tensor buffer[],
                             u32 buffer_length) {
  if (!buffer)
    return nullptr;
  if (buffer_length < N_LINEAR_KERNELS)
    return nullptr;

  buffer[0] = layer->weight;
  buffer[1] = layer->bias;

  return buffer;
}
