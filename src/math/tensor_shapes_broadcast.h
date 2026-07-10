#ifndef TENSOR_SHAPES_BROADCAST
#define TENSOR_SHAPES_BROADCAST
#include "tensor.h"

static inline bool
tensor_shapes_broadcast_from_shape(u32 t_ndims, const u32 *t_shape, u32 s_ndims,
                                   const u32 *s_shape, u32 *outshape);

static inline bool tensor_shapes_broadcast(const Tensor tensor_a,
                                           const Tensor tensor_b,
                                           u32 *outshape) {

  auto t_ndims = tensor_ndims(tensor_a);
  auto s_ndims = tensor_ndims(tensor_b);

  auto t_shape = tensor_shape(tensor_a);
  auto s_shape = tensor_shape(tensor_b);

  return tensor_shapes_broadcast_from_shape(t_ndims, t_shape, s_ndims, s_shape,
                                            outshape);
}

static inline bool tensor_shapes_broadcast_from_shape(const u32 t_ndims,
                                                      const u32 *t_shape,
                                                      const u32 s_ndims,
                                                      const u32 *s_shape,
                                                      u32 *outshape) {

  if (t_shape == NULL || s_shape == NULL)
    return false;

  u32 indx_t = t_ndims;
  u32 indx_s = s_ndims;
  u32 indx_o = indx_t > indx_s ? indx_t : indx_s;

  u32 *shape_o = outshape;

  for (; indx_t > 0 && indx_s > 0 && indx_o > 0;) {
    --indx_t;
    --indx_s;
    --indx_o;

    if (t_shape[indx_t] == s_shape[indx_s]) {
      shape_o[indx_o] = t_shape[indx_t];
    } else if (s_shape[indx_s] == 1) {
      shape_o[indx_o] = t_shape[indx_t];
    } else if (t_shape[indx_t] == 1) {
      shape_o[indx_o] = s_shape[indx_s];
    } else {
      return false;
    }
  }

  while (indx_t > 0) {
    --indx_t;
    --indx_o;
    shape_o[indx_o] = t_shape[indx_t];
  }

  while (indx_s > 0) {
    --indx_s;
    --indx_o;
    shape_o[indx_o] = s_shape[indx_s];
  }

  return true;
}

#endif
