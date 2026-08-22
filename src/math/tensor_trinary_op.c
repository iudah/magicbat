#include "tensor_trinary_op.h"
#include "tensor.h"
#include "tensor_odometer.h"
#include "tensor_prot.h"
#include "tensor_shapes_broadcast.h"
#include "tensor_shapes_equal.h"
#include "type_alias.h"
#include <stdint.h>

Tensor
tensor_same_shape_trinary(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c,
                          float (*operation_callback)(float, float, float));
Tensor tensor_broadcast_trinary(Tensor tensor_a, Tensor tensor_b,
                                Tensor tensor_c, u32 o_ndims, u32 *o_shape,
                                float (*operation_callback)(float, float,
                                                            float));
Tensor
tensor_length_one_a_trinary(Tensor tensor_b, Tensor tensor_c, f32 float_a,
                            float (*operation_callback)(float, float, float));
Tensor
tensor_length_one_b_trinary(Tensor tensor_a, Tensor tensor_c, f32 float_b,
                            float (*operation_callback)(float, float, float));
Tensor
tensor_length_one_c_trinary(Tensor tensor_a, Tensor tensor_b, f32 float_c,
                            float (*operation_callback)(float, float, float));

Tensor tensor_trinary_op(const Tensor tensor_a, const Tensor tensor_b,
                         const Tensor tensor_c,
                         float (*operation_callback)(float, float, float)) {
  TASSERT(tensor_a && tensor_b && tensor_c && operation_callback &&
          "Null tensor or operator.");

  if (!tensor_a || !tensor_b || !tensor_c)
    return nullptr;

  if (tensor_a->is_contiguous && tensor_b->is_contiguous &&
      tensor_c->is_contiguous && tensor_shapes_equal(tensor_a, tensor_b) &&
      tensor_shapes_equal(tensor_a, tensor_c))
    return tensor_same_shape_trinary(tensor_a, tensor_b, tensor_c,
                                     operation_callback);
  if (tensor_num_elements(tensor_b) == 1) {
    return tensor_length_one_b_trinary(tensor_a, tensor_b,
                                       tensor_get(tensor_c, (u32[MAX_DIMS]){0}),
                                       operation_callback);
  }
  if (tensor_num_elements(tensor_c) == 1) {
    return tensor_length_one_c_trinary(tensor_a, tensor_b,
                                       tensor_get(tensor_c, (u32[MAX_DIMS]){0}),
                                       operation_callback);
  }
  /*
    if (tensor_num_elements(tensor_a) == 1)
      tensor_length_one_a_binary(tensor_a, tensor_b, alpha, operation_callback);

    if (tensor_num_elements(tensor_b) == 1)
      tensor_length_one_b_binary(tensor_a, tensor_b, alpha, operation_callback);
  */
  u32 o_ndims =
      tensor_a->ndims > tensor_b->ndims ? tensor_a->ndims : tensor_b->ndims;
  o_ndims = o_ndims > tensor_c->ndims ? o_ndims : tensor_c->ndims;

  u32 t_shape[MAX_DIMS];
  u32 o_shape[MAX_DIMS];

  if (tensor_shapes_broadcast(tensor_a, tensor_b, t_shape) &&
      tensor_shapes_broadcast_from_shape(
          tensor_a->ndims > tensor_b->ndims ? tensor_a->ndims : tensor_b->ndims,
          t_shape, tensor_c->ndims, tensor_c->shape, o_shape)) {
    return tensor_broadcast_trinary(tensor_a, tensor_b, tensor_c, o_ndims,
                                    o_shape, operation_callback);
  }
  return nullptr;
}

Tensor
tensor_same_shape_trinary(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c,
                          float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_a->data->data[i];
    auto val_b = tensor_b->data->data[i];
    auto val_c = tensor_c->data->data[i];
    res->data->data[i] = operation_callback(val_a, val_b, val_c);
  }

  return res;
}
Tensor tensor_broadcast_trinary(Tensor tensor_a, Tensor tensor_b,
                                Tensor tensor_c, u32 o_ndims, u32 *o_shape,
                                float (*operation_callback)(float, float,
                                                            float)) {

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
  u32 u_nindx[MAX_DIMS];
  u32 flat = 0;

  do {
    for (u32 i = res->ndims, j = tensor_a->ndims; i-- > 0 && j-- > 0;) {
      t_nindx[j] = tensor_a->shape[j] > 1 ? index[i] : 0;
    }

    for (u32 i = res->ndims, k = tensor_b->ndims; i-- > 0 && k-- > 0;) {
      s_nindx[k] = tensor_b->shape[k] > 1 ? index[i] : 0;
    }

    for (u32 i = res->ndims, ki = tensor_c->ndims; i-- > 0 && ki-- > 0;) {
      u_nindx[ki] = tensor_c->shape[ki] > 1 ? index[i] : 0;
    }

    auto t_val = tensor_get(tensor_a, t_nindx);
    auto s_val = tensor_get(tensor_b, s_nindx);
    auto u_val = tensor_get(tensor_c, u_nindx);

    auto r_val = operation_callback(t_val, s_val, u_val);

    res->data->data[flat] = r_val;
    ++flat;

  } while (tensor_odometer_next(index, o_ndims, o_shape));
  tensor_odometer_destroy(index);
  return res;
}

Tensor tensor_same_shape_length_one_c_trinary(
    Tensor tensor_a, Tensor tensor_b, f32 alpha,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_a->data->data[i];
    auto val_b = tensor_b->data->data[i];
    res->data->data[i] = operation_callback(val_a, val_b, alpha);
  }

  return res;
}

Tensor tensor_length_one_a_length_one_c_trinary(
    f32 val_a, Tensor tensor_b, f32 float_c,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_b), tensor_shape(tensor_b));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor_b->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto val_b = tensor_b->data->data[i];
      res->data->data[i] = operation_callback(val_a, val_b, float_c);
    }

  } else {
    u32 index[MAX_DIMS] = {0};

    for (u32 i = 0; i < nelement; ++i) {
      auto val_b = tensor_get(tensor_b, index);
      res->data->data[i] = operation_callback(val_a, val_b, float_c);
      tensor_odometer_next(index, tensor_b->ndims, tensor_b->shape);
    }
  }
  return res;
}

Tensor tensor_length_one_b_length_one_c_trinary(
    Tensor tensor_a, f32 val_b, f32 float_c,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor_a->is_contiguous) {

    for (u32 i = 0; i < nelement; ++i) {
      auto val_a = tensor_a->data->data[i];
      res->data->data[i] = operation_callback(val_a, val_b, float_c);
    }
  } else {
    u32 index[MAX_DIMS] = {0};

    for (u32 i = 0; i < nelement; ++i) {
      auto val_a = tensor_get(tensor_a, index);
      res->data->data[i] = operation_callback(val_a, val_b, float_c);
      tensor_odometer_next(index, tensor_a->ndims, tensor_a->shape);
    }
  }
  return res;
}

Tensor tensor_broadcast_length_one_c_trinary(
    Tensor tensor_a, Tensor tensor_b, u32 o_ndims, u32 *o_shape, f32 float_c,
    float (*operation_callback)(float, float, float)) {

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

    auto r_val = operation_callback(t_val, s_val, float_c);

    res->data->data[flat] = r_val;
    ++flat;

  } while (tensor_odometer_next(index, o_ndims, o_shape));
  tensor_odometer_destroy(index);
  return res;
}

Tensor tensor_length_one_c_trinary(const Tensor tensor_a, const Tensor tensor_b,
                                   f32 float_c,
                                   float (*operation_callback)(float, float,
                                                               float)) {
  TASSERT(tensor_a && tensor_b && tensor_c && operation_callback &&
          "Null tensor or operator.");

  if (!tensor_a || !tensor_b)
    return nullptr;

  bool contiguous_duo = tensor_a->is_contiguous && tensor_b->is_contiguous;

  if (contiguous_duo && tensor_shapes_equal(tensor_a, tensor_b))
    return tensor_same_shape_length_one_c_trinary(tensor_a, tensor_b, float_c,
                                                  operation_callback);

  if (tensor_num_elements(tensor_a) == 1)
    tensor_length_one_a_length_one_c_trinary(
        tensor_get(tensor_a, (u32[MAX_DIMS]){0}), tensor_b, float_c,
        operation_callback);

  if (tensor_num_elements(tensor_b) == 1)
    tensor_length_one_b_length_one_c_trinary(
        tensor_a, tensor_get(tensor_b, (u32[MAX_DIMS]){0}), float_c,
        operation_callback);

  u32 o_ndims =
      tensor_a->ndims > tensor_b->ndims ? tensor_a->ndims : tensor_b->ndims;
  u32 o_shape[MAX_DIMS];

  if (tensor_shapes_broadcast(tensor_a, tensor_b, o_shape)) {
    return tensor_broadcast_length_one_c_trinary(
        tensor_a, tensor_b, o_ndims, o_shape, float_c, operation_callback);
  }
  return nullptr;
}

//---
Tensor tensor_same_shape_length_one_b_trinary(
    Tensor tensor_a, Tensor tensor_c, f32 float_b,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_a), tensor_shape(tensor_a));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_a->data->data[i];
    auto val_c = tensor_c->data->data[i];
    res->data->data[i] = operation_callback(val_a, float_b, val_c);
  }

  return res;
}

Tensor tensor_length_one_a_length_one_b_trinary(
    f32 val_a, Tensor tensor_c, f32 float_b,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_c), tensor_shape(tensor_c));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  if (tensor_c->is_contiguous) {
    for (u32 i = 0; i < nelement; ++i) {
      auto val_c = tensor_c->data->data[i];
      res->data->data[i] = operation_callback(val_a, float_b, val_c);
    }

  } else {
    u32 index[MAX_DIMS] = {0};

    for (u32 i = 0; i < nelement; ++i) {
      auto val_c = tensor_get(tensor_c, index);
      res->data->data[i] = operation_callback(val_a, float_b, val_c);
      tensor_odometer_next(index, tensor_c->ndims, tensor_c->shape);
    }
  }
  return res;
}

Tensor tensor_broadcast_length_one_b_trinary(
    Tensor tensor_a, Tensor tensor_c, u32 o_ndims, u32 *o_shape, f32 float_b,
    float (*operation_callback)(float, float, float)) {

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

    for (u32 i = res->ndims, k = tensor_c->ndims; i-- > 0 && k-- > 0;) {
      s_nindx[k] = tensor_c->shape[k] > 1 ? index[i] : 0;
    }

    auto t_val = tensor_get(tensor_a, t_nindx);
    auto s_val = tensor_get(tensor_c, s_nindx);

    auto r_val = operation_callback(t_val, float_b, s_val);

    res->data->data[flat] = r_val;
    ++flat;

  } while (tensor_odometer_next(index, o_ndims, o_shape));
  tensor_odometer_destroy(index);
  return res;
}

Tensor tensor_length_one_b_trinary(const Tensor tensor_a, const Tensor tensor_c,
                                   f32 float_b,
                                   float (*operation_callback)(float, float,
                                                               float)) {
  TASSERT(tensor_a && tensor_b && tensor_c && operation_callback &&
          "Null tensor or operator.");

  if (!tensor_a || !tensor_c)
    return nullptr;

  bool contiguous_duo = tensor_a->is_contiguous && tensor_c->is_contiguous;

  if (contiguous_duo && tensor_shapes_equal(tensor_a, tensor_c))
    return tensor_same_shape_length_one_b_trinary(tensor_a, tensor_c, float_b,
                                                  operation_callback);

  if (tensor_num_elements(tensor_a) == 1)
    tensor_length_one_a_length_one_b_trinary(
        tensor_get(tensor_a, (u32[MAX_DIMS]){0}), tensor_c, float_b,
        operation_callback);

  if (tensor_num_elements(tensor_c) == 1)
    tensor_length_one_b_length_one_c_trinary(
        tensor_a, tensor_get(tensor_a, (u32[MAX_DIMS]){0}), float_b,
        operation_callback);

  u32 o_ndims =
      tensor_a->ndims > tensor_c->ndims ? tensor_a->ndims : tensor_c->ndims;
  u32 o_shape[MAX_DIMS];

  if (tensor_shapes_broadcast(tensor_a, tensor_c, o_shape)) {
    return tensor_broadcast_length_one_b_trinary(
        tensor_a, tensor_c, o_ndims, o_shape, float_b, operation_callback);
  }
  return nullptr;
}

//---
Tensor tensor_same_shape_length_one_a_trinary(
    Tensor tensor_b, Tensor tensor_c, f32 float_a,
    float (*operation_callback)(float, float, float)) {
  Tensor res = tensor_new(tensor_ndims(tensor_b), tensor_shape(tensor_b));
  if (res == nullptr)
    return nullptr;

  auto nelement = tensor_num_elements(res);

  for (u32 i = 0; i < nelement; ++i) {
    auto val_a = tensor_b->data->data[i];
    auto val_c = tensor_c->data->data[i];
    res->data->data[i] = operation_callback(val_a, float_a, val_c);
  }

  return res;
}

Tensor tensor_broadcast_length_one_a_trinary(
    Tensor tensor_b, Tensor tensor_c, u32 o_ndims, u32 *o_shape, f32 float_a,
    float (*operation_callback)(float, float, float)) {

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
    for (u32 i = res->ndims, j = tensor_b->ndims; i-- > 0 && j-- > 0;) {
      t_nindx[j] = tensor_b->shape[j] > 1 ? index[i] : 0;
    }

    for (u32 i = res->ndims, k = tensor_c->ndims; i-- > 0 && k-- > 0;) {
      s_nindx[k] = tensor_c->shape[k] > 1 ? index[i] : 0;
    }

    auto t_val = tensor_get(tensor_b, t_nindx);
    auto s_val = tensor_get(tensor_c, s_nindx);

    auto r_val = operation_callback(t_val, float_a, s_val);

    res->data->data[flat] = r_val;
    ++flat;

  } while (tensor_odometer_next(index, o_ndims, o_shape));
  tensor_odometer_destroy(index);
  return res;
}

Tensor tensor_length_one_a_trinary(const Tensor tensor_b, const Tensor tensor_c,
                                   f32 float_a,
                                   float (*operation_callback)(float, float,
                                                               float)) {
  TASSERT(tensor_a && tensor_b && tensor_c && operation_callback &&
          "Null tensor or operator.");

  if (!tensor_b || !tensor_c)
    return nullptr;

  bool contiguous_duo = tensor_b->is_contiguous && tensor_c->is_contiguous;

  if (contiguous_duo && tensor_shapes_equal(tensor_b, tensor_c))
    return tensor_same_shape_length_one_a_trinary(tensor_b, tensor_c, float_a,
                                                  operation_callback);

  if (tensor_num_elements(tensor_b) == 1)
    tensor_length_one_a_length_one_b_trinary(
        float_a, tensor_c, tensor_get(tensor_b, (u32[MAX_DIMS]){0}),
        operation_callback);

  if (tensor_num_elements(tensor_c) == 1)
    tensor_length_one_a_length_one_c_trinary(
        float_a, tensor_b, tensor_get(tensor_b, (u32[MAX_DIMS]){0}),
        operation_callback);

  u32 o_ndims =
      tensor_b->ndims > tensor_c->ndims ? tensor_b->ndims : tensor_c->ndims;
  u32 o_shape[MAX_DIMS];

  if (tensor_shapes_broadcast(tensor_b, tensor_c, o_shape)) {
    return tensor_broadcast_length_one_a_trinary(
        tensor_b, tensor_c, o_ndims, o_shape, float_a, operation_callback);
  }
  return nullptr;
}
