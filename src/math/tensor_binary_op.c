#include "tensor_binary_op.h"
#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_odometer.h"
#include "tensor_shapes_broadcast.h"
#include "tensor_shapes_equal.h"
#include <stdint.h>

Tensor tensor_binary_op(const Tensor t, const Tensor s,
                        float (*op)(float, float)) {
  if (!t || !s)
    return NULL;

  if (tensor_shapes_equal(t, s)) {
    Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
    if (res == NULL)
      return NULL;

    auto nelement = tensor_num_elements(res);

    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data->data[i];
      auto b = s->data->data[i];
      res->data->data[i] = op(a, b);
    }

    return res;
  }

  if (tensor_num_elements(t) == 1) {
    Tensor res = tensor_new(tensor_ndims(s), tensor_shape(s));
    if (res == NULL)
      return NULL;

    auto nelement = tensor_num_elements(res);

    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data->data[0];
      auto b = s->data->data[i];
      res->data->data[i] = op(a, b);
    }
    return res;
  }

  if (tensor_num_elements(s) == 1) {
    Tensor res = tensor_new(tensor_ndims(t), tensor_shape(t));
    if (res == NULL)
      return NULL;

    auto nelement = tensor_num_elements(res);

    for (u32 i = 0; i < nelement; ++i) {
      auto a = t->data->data[i];
      auto b = s->data->data[0];
      res->data->data[i] = op(a, b);
    }
    return res;
  }

  u32 o_ndims = t->ndims > s->ndims ? t->ndims : s->ndims;
  u32 *t_stride = tmalloc(o_ndims * 4 * sizeof(u32));
  u32 *s_stride = t_stride + o_ndims;
  u32 *o_stride = s_stride + o_ndims;
  u32 *o_shape = o_stride + o_ndims;
  if (tensor_shapes_broadcast(t, s, o_shape, o_stride, t_stride, s_stride)) {
    Tensor res = tensor_new(o_ndims, o_shape);
    if (res == NULL) {
      return NULL;
    }

    u32 *index = tensor_odometer_new(o_ndims);
    if (index == NULL) {
      tensor_destroy(res);
      res = NULL;
      return NULL;
    }
    do {
      u32 o_indx = 0;
      u32 t_indx = 0;
      u32 s_indx = 0;

      for (u32 i = 0; i < o_ndims; ++i) {
        o_indx += o_stride[i] * index[i];
        t_indx += t_stride[i] * index[i];
        s_indx += s_stride[i] * index[i];
      }

      res->data->data[o_indx] =
          op(t->data->data[t_indx], s->data->data[s_indx]);
    } while (tensor_odometer_next(index, o_ndims, o_shape));
    tensor_odometer_destroy(index);
    tfree(t_stride);
    return res;
  }
  tfree(t_stride);
  return NULL;
}
