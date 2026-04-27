#ifndef TENSOR_SHAPES_EQUAL
#define TENSOR_SHAPES_EQUAL
#include "../../include/tensor.h"

static inline bool tensor_shapes_equal_from_shape(const u32 t_ndims,
                                                  const u32 *t_shape,
                                                  const u32 s_ndims,
                                                  const u32 *s_shape);

static inline bool tensor_shapes_equal(const Tensor t, const Tensor s) {
  auto t_size = tensor_num_elements(t);
  auto s_size = tensor_num_elements(s);

  if (t_size != s_size)
    return false;

  auto t_ndims = tensor_ndims(t);
  auto s_ndims = tensor_ndims(s);

  auto t_shape = tensor_shape(t);
  auto s_shape = tensor_shape(s);

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

  u32 i = t_ndims;
  u32 j = s_ndims;
  for (; i > 0 && j > 0;) {
    --i;
    --j;

    if (t_shape[i] != s_shape[j])
      return false;
  }

  return true;
}
#endif
