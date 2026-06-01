#ifndef TENSOR_SHAPES_EQUAL
#define TENSOR_SHAPES_EQUAL
#include "../../include/tensor.h"

static inline bool tensor_shapes_equal_from_shape(u32 t_ndims,
                                                  const u32 *t_shape,
                                                  u32 s_ndims,
                                                  const u32 *s_shape);

static inline bool tensor_shapes_equal(const Tensor var_a, const Tensor var_b) {
  auto t_size = tensor_num_elements(var_a);
  auto s_size = tensor_num_elements(var_b);

  if (t_size != s_size)
    return false;

  auto t_ndims = tensor_ndims(var_a);
  auto s_ndims = tensor_ndims(var_b);

  auto t_shape = tensor_shape(var_a);
  auto s_shape = tensor_shape(var_b);

  return tensor_shapes_equal_from_shape(t_ndims, t_shape, s_ndims, s_shape);
}

static inline bool tensor_shapes_equal_from_shape(const u32 t_ndims,
                                                  const u32 *t_shape,
                                                  const u32 s_ndims,
                                                  const u32 *s_shape) {

  if (t_ndims != s_ndims)
    return false;
  if (t_ndims == 0)
    return false;

  if (t_shape == NULL || s_shape == NULL)
    return false;

  u32 t_indx = t_ndims;
  u32 s_indx = s_ndims;
  for (; t_indx > 0 && s_indx > 0;) {
    --t_indx;
    --s_indx;

    if (t_shape[t_indx] != s_shape[s_indx])
      return false;
  }

  return true;
}
#endif
