#ifndef TENSOR_SHAPES_BROADCAST
#define TENSOR_SHAPES_BROADCAST
#include "../../include/tensor.h"

static inline bool tensor_shapes_broadcast(const Tensor t, const Tensor s,
                                           u32 *outshape, u32 *outstride,
                                           u32 *tstride, u32 *sstride) {

  auto t_ndims = tensor_ndims(t);
  auto s_ndims = tensor_ndims(s);

  auto t_shape = tensor_shape(t);
  auto s_shape = tensor_shape(s);

  if (t_shape == NULL || s_shape == NULL)
    return false;

  u32 i = t_ndims;
  u32 j = s_ndims;
  u32 max = i > j ? i : j;

  u32 length_o = 1;
  u32 length_s = 1;
  u32 length_t = 1;

  u32 *shape_o = outshape;
  u32 *stride_o = outstride;
  u32 *stride_t = tstride;
  u32 *stride_s = sstride;

  while (i > 0 && j > 0) {
    --i;
    --j;
    --max;
    if (t_shape[i] == s_shape[j]) {
      stride_s[max] = length_s;
      stride_t[max] = length_t;
      stride_o[max] = length_o;
      shape_o[max] = t_shape[i];
      length_t *= t_shape[i];
      length_s *= s_shape[j];
      length_o *= t_shape[i];
    } else if (s_shape[j] == 1) {
      stride_s[max] = 0;
      stride_t[max] = length_t;
      stride_o[max] = length_o;
      shape_o[max] = t_shape[i];
      length_t *= t_shape[i];
      length_o *= t_shape[i];
    } else if (t_shape[i] == 1) {
      stride_s[max] = length_s;
      stride_t[max] = 0;
      stride_o[max] = length_o;
      shape_o[max] = s_shape[j];
      length_s *= s_shape[j];
      length_o *= s_shape[j];
    } else {
      return false;
    }
  }

  while (i > 0) {
    --i;
    --max;
    stride_s[max] = 0;
    stride_t[max] = length_t;
    stride_o[max] = length_o;
    shape_o[max] = t_shape[i];
    length_t *= t_shape[i];
    length_o *= t_shape[i];
  }

  while (j > 0) {
    --j;
    --max;
    stride_s[max] = length_s;
    stride_t[max] = 0;
    stride_o[max] = length_o;
    shape_o[max] = s_shape[j];
    length_s *= s_shape[j];
    length_o *= s_shape[j];
  }

  return true;
}

#endif
