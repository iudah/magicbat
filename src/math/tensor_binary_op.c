#include "tensor_binary_op.h"
#include "../../include/adt/tensor/tensor_prot.h"
#include "../../include/tensor.h"
#include "tensor_odometer.h"
#include "tensor_shapes_broadcast.h"
#include "tensor_shapes_equal.h"
#include <stdint.h>

Tensor tensor_same_shape_binary(Tensor tensor_a, Tensor tensor_b,
                                float (*operation_callback)(float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_a->data->data[i];
    auto val_b = tensor_b->data->data[i];
    res->data->data[i] = operation_callback(val_a, val_b);
  }

  return res;
}

Tensor tensor_length_one_a_binary(Tensor tensor_a, Tensor tensor_b,
                                  float (*operation_callback)(float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_b), tensor_shape(tensor_b));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_a->data->data[0];
    auto val_b = tensor_b->data->data[i];
    res->data->data[i] = operation_callback(val_a, val_b);
  }
  return res;
}

Tensor tensor_length_one_b_binary(Tensor tensor_a, Tensor tensor_b,
                                  float (*operation_callback)(float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor_a->is_contiguous && tensor_b->is_contiguous) {
    auto val_b = tensor_b->data->data[0];

    for (u32 i = 0; i < nelement; ++i) {
      auto val_a = tensor_a->data->data[i];
      res->data->data[i] = operation_callback(val_a, val_b);
    }
  } else {
    auto val_b = tensor_get(tensor_b, (u32[MAX_DIMS]){0});
    auto index = tensor_odometer_new(tensor_a->ndims);

    for (u32 i = 0; i < nelement; ++i) {
      auto val_a = tensor_get(tensor_a, index);
      res->data->data[i] = operation_callback(val_a, val_b);
      tensor_odometer_next(index, tensor_a->ndims, tensor_a->shape);
    }

    tensor_odometer_destroy(index);
  }
  return res;
}

Tensor tensor_broadcast_binary(Tensor tensor_a, Tensor tensor_b, u32 o_ndims,
                               u32 *o_shape,
                               float (*operation_callback)(float, float)) {

  Tensor res = tensor_new(o_ndims, o_shape);
  if (res == nullptr) {
    return nullptr;
  }

  u32 *index = tensor_odometer_new(o_ndims);
  if (index == nullptr) {
    tensor_destroy(res);
    res = nullptr;
    return nullptr;
  }

  u32 t_nindx[MAX_DIMS];
  u32 s_nindx[MAX_DIMS];
  u32 flat = 0;

  do {
    for (u32 i = res->ndims, j = tensor_a->ndims; i-- > 0 && j-- > 0;) {
      t_nindx[j] = tensor_a->shape[j] > 1 ? index[i] : 0;
    }

    for (u32 i = res->ndims, k = tensor_b->ndims; i-- > 0 && k-- > 0;) {
      s_nindx[k] = tensor_b->shape[k] > 1 ? index[i] : 0;
    }

    auto t_val = tensor_get(tensor_a, t_nindx);
    auto s_val = tensor_get(tensor_b, s_nindx);

    auto r_val = operation_callback(t_val, s_val);

    res->data->data[flat] = r_val;
    ++flat;

  } while (tensor_odometer_next(index, o_ndims, o_shape));
  tensor_odometer_destroy(index);
  return res;
}

Tensor tensor_binary_op(const Tensor tensor_a, const Tensor tensor_b,
                        float (*operation_callback)(float, float)) {
  TASSERT(tensor_a && tensor_b && operation_callback &&
          "Null tensor or operator.");

  if (!tensor_a || !tensor_b)
    return nullptr;

  if (tensor_shapes_equal(tensor_a, tensor_b))
    return tensor_same_shape_binary(tensor_a, tensor_b, operation_callback);

  if (tensor_num_elements(tensor_a) == 1)
    tensor_length_one_a_binary(tensor_a, tensor_b, operation_callback);

  if (tensor_num_elements(tensor_b) == 1)
    tensor_length_one_b_binary(tensor_a, tensor_b, operation_callback);

  u32 o_ndims =
      tensor_a->ndims > tensor_b->ndims ? tensor_a->ndims : tensor_b->ndims;
  u32 o_shape[MAX_DIMS];

  if (tensor_shapes_broadcast(tensor_a, tensor_b, o_shape)) {
    return tensor_broadcast_binary(tensor_a, tensor_b, o_ndims, o_shape,
                                   operation_callback);
  }
  return nullptr;
}

bool tensor_binary_op_inplace(Tensor restrict tensor_a,
                              const Tensor restrict tensor_b,
                              float (*operation_callback)(float, float)) {
  if (!tensor_a || !tensor_b)
    return false;

  if (tensor_shapes_equal(tensor_a, tensor_b)) {
    Tensor res = tensor_a;

    auto nelement = tensor_num_elements(res);

    for (u32 i = 0; i < nelement; ++i) {
      auto val_a = tensor_a->data->data[i];
      auto val_b = tensor_b->data->data[i];
      res->data->data[i] = operation_callback(val_a, val_b);
    }

    return true;
  }

  if (tensor_shape_is_broadcast(tensor_b->ndims, tensor_b->shape,
                                tensor_a->ndims, tensor_a->shape)) {

    Tensor res = tensor_a;
    u32 index[MAX_DIMS] = {0};
    u32 *odometer = tensor_odometer_new(tensor_a->ndims);
    auto nelement = tensor_num_elements(res);

    for (u32 i = 0; i < nelement; ++i) {
      for (u32 j = tensor_a->ndims, k = tensor_b->ndims; j-- > 0 && k-- > 0;) {
        index[k] = tensor_b->shape[k] == 1 ? 0 : odometer[j];
      }

      auto val_a = tensor_a->data->data[i];
      auto val_b = tensor_get(tensor_b, index);
      res->data->data[i] = operation_callback(val_a, val_b);

      tensor_odometer_next(odometer, tensor_a->ndims, tensor_a->shape);
    }
    tensor_odometer_destroy(odometer);

    return true;
  }
  return false;
}

bool tensor_binary_op_scalar_inplace(Tensor restrict tensor_a, const f32 scalar,
                                     float (*operation_callback)(float,
                                                                 float)) {
  if (!tensor_a)
    return false;

  Tensor res = tensor_a;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto value = tensor_a->data->data[i];
    res->data->data[i] = operation_callback(value, scalar);
  }

  return true;
}
