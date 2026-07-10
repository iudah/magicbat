#include "tensor.h"
#include <math.h>

Tensor tensor_positional_encoding(const u32 seq, const u32 dmodel) {
  auto pos_enc = tensor_new(2, (u32[]){seq, dmodel});

#define PE_CONST 10000.f

  for (u32 pos = 0; pos < seq; ++pos) {
    for (u32 i = 0; i < dmodel; i += 2) {
      auto freq = pos * powf(PE_CONST, -(f32)i / (f32)dmodel);
      pos_enc->data->data[(pos * dmodel) + i] = sinf(freq);
      pos_enc->data->data[(pos * dmodel) + i + 1] = cosf(freq);
    }
  }

  return pos_enc;
}
