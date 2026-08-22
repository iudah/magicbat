#include "tensor.h"
#include "tensor_ndim_flat_index.h"
#include "tensor_odometer.h"
#include <math.h>
#include <stdint.h>

#define EPS (1e-6)
static void softmax_last_axis(u32 outer, u32 target, const f32 *restrict input,
                              f32 *restrict output) {
  for (u32 i = 0; i < outer; ++i) {
    f32 sum = 0;
    f32 max = -INFINITY;
    for (u32 j = 0; j < target; ++j) {
      f32 score = input[(i * target) + j];
      f32 new_max = fmaxf(max, score);
      f32 probability = expf(score - new_max);
      f32 diff = max - new_max;
      f32 alpha = expf(diff);
      sum = (sum * alpha) + probability;
      max = new_max;
    }
    for (u32 j = 0; j < target; ++j) {
      f32 score = input[(i * target) + j];
      f32 probability = expf(score - max);
      output[(i * target) + j] = probability / sum;
    }
  }
}
static void softmax_not_last_axis(u32 outer, u32 target, u32 inner,
                                  const f32 *restrict input,
                                  f32 *restrict output) {

  f32 *sum_array = tmalloc(inner * sizeof(*sum_array));
  f32 *max_array = tmalloc(inner * sizeof(*sum_array));
  for (u32 i = 0; i < outer; ++i) {
    for (u32 k = 0; k < inner; ++k) {
      sum_array[k] = 0;
      max_array[k] = -INFINITY;
    }
    for (u32 j = 0; j < target; ++j) {
      for (u32 k = 0; k < inner; ++k) {
        f32 score = input[((i * target + j) * inner) + k];
        f32 max = max_array[k];
        f32 sum = sum_array[k];
        f32 new_max = fmaxf(max, score);
        f32 probability = expf(score - new_max);
        f32 diff = max - new_max;
        f32 alpha = expf(diff);
        sum_array[k] = (sum * alpha) + probability;
        max_array[k] = new_max;
      }
    }
    for (u32 j = 0; j < target; ++j) {
      for (u32 k = 0; k < inner; ++k) {
        f32 score = input[((i * target + j) * inner) + k];
        f32 max = max_array[k];
        f32 sum = sum_array[k];
        f32 probability = expf(score - max);
        output[((i * target + j) * inner) + k] = probability / sum;
      }
    }
  }
  tfree(sum_array);
  tfree(max_array);
}
static void softmax_not_contiguous_axis(u32 outer_dim, const u32 *outer,
                                        u32 axis, u32 target, u32 inner_dim,
                                        const u32 *inner,
                                        const Tensor restrict tensor,
                                        f32 *restrict output) {
  const f32 *restrict input = tensor->data->data;
  u32 index[MAX_DIMS] = {0};
  u32 inner_len = 1;
  for (u32 i = 0; i < inner_dim; ++i) {
    inner_len *= inner[i];
  }

  f32 *sum_array = tmalloc(inner_len * sizeof(*sum_array));
  f32 *max_array = tmalloc(inner_len * sizeof(*sum_array));

  do {
    for (u32 k = 0; k < inner_len; ++k) {
      sum_array[k] = 0;
      max_array[k] = -INFINITY;
    }

    for (u32 j = 0; j < target; ++j) {
      u32 inner_flat = 0;
      tensor_odometer_reset(inner_dim, index + axis + 1);
      do {
        index[axis] = j;
        auto flat = tensor_ndim_flat_index(tensor, index);
        f32 score = input[flat];
        f32 max = max_array[inner_flat];
        f32 sum = sum_array[inner_flat];
        f32 new_max = fmaxf(max, score);
        f32 probability = expf(score - new_max);
        f32 diff = max - new_max;
        f32 alpha = expf(diff);
        sum_array[inner_flat] = (sum * alpha) + probability;
        max_array[inner_flat] = new_max;
        ++inner_flat;
      } while (tensor_odometer_next(index + axis + 1, inner_dim, inner));
    }
    for (u32 j = 0; j < target; ++j) {
      u32 inner_flat = 0;
      tensor_odometer_reset(inner_dim, index + axis + 1);
      do {
        index[axis] = j;
        auto flat = tensor_ndim_flat_index(tensor, index);
        f32 score = input[flat];
        f32 max = max_array[inner_flat];
        f32 sum = sum_array[inner_flat];
        f32 new_max = fmaxf(max, score);
        f32 probability = expf(score - new_max);
        output[flat] = probability / sum;
        ++inner_flat;
      } while (tensor_odometer_next(index + axis + 1, inner_dim, inner));
    }
  } while (tensor_odometer_next(index, outer_dim, outer));
}

Tensor tensor_softmax(const Tensor tensor, i32 axis) {
  if (!tensor)
    return nullptr;

  u32 reduce_axis = axis < 0 ? tensor->ndims + axis : axis;
  if (reduce_axis < 0)
    return nullptr;

  auto res = tensor_new(tensor->ndims, tensor->shape);
  if (!tensor->is_contiguous) {
    softmax_not_contiguous_axis(
        reduce_axis, tensor->shape, reduce_axis, tensor->shape[reduce_axis],
        tensor->ndims - reduce_axis - 1, tensor->shape + reduce_axis + 1,
        tensor, res->data->data);

    return res;
  }

  u32 outer = 1;
  for (u32 i = 0; i < reduce_axis; ++i) {
    outer *= tensor->shape[i];
  }

  u32 inner = 1;
  for (u32 i = reduce_axis + 1; i < tensor->ndims; ++i) {
    inner *= tensor->shape[i];
  }

  if (reduce_axis + 1 == tensor->ndims) {
    softmax_last_axis(outer, tensor->shape[reduce_axis], tensor->data->data,
                      res->data->data);
  } else {
    softmax_not_last_axis(outer, tensor->shape[reduce_axis], inner,
                          tensor->data->data, res->data->data);
  }

  return res;
}

static void dsoftmax_last_axis(u32 outer, u32 target,
                               const f32 *restrict softmax,
                               const f32 *restrict grad, f32 *restrict output) {
  // So dsoftmaxdx[i,j]=S[i,j]*(dS[i,j]-\sum_{k=0}^{J-1}{dS[i,k]*S[i,k]}) Where
  // shape(S)=(I,J)
  for (u32 i = 0; i < outer; ++i) {
    f32 dot = 0;
    for (u32 k = 0; k < target; ++k) {
      dot += grad[(i * target) + k] * softmax[(i * target) + k];
    }
    for (u32 j = 0; j < target; ++j) {
      output[(i * target) + j] =
          softmax[(i * target) + j] * (grad[(i * target) + j] - dot);
    }
  }
}
static void dsoftmax_not_last_axis(u32 outer, u32 target, u32 inner,
                                   const f32 *restrict softmax,
                                   const f32 *restrict grad,
                                   f32 *restrict output) {
  // So dsoftmaxdx[i,j]=S[i,j]*(dS[i,j]-\sum_{k=0}^{J-1}{dS[i,k]*S[i,k]}) Where
  // shape(S)=(I,J)
  f32 *dot_array = tmalloc(inner * sizeof(*dot_array));
  for (u32 i = 0; i < outer; ++i) {
    for (u32 k = 0; k < target; ++k) {
      dot_array[k] = 0;
    }
    for (u32 k = 0; k < target; ++k) {
      for (u32 j = 0; j < inner; ++j) {
        dot_array[k] += grad[((i * target + k) * inner) + j] *
                        softmax[((i * target + k) * inner) + j];
      }
    }
    for (u32 j = 0; j < target; ++j) {
      for (u32 k = 0; k < inner; ++k) {
        output[((i * target + j) * inner) + k] =
            softmax[((i * target + j) * inner) + k] *
            (grad[((i * target + j) * inner) + k] - dot_array[k]);
      }
    }
  }
}

Tensor tensor_softmax_backward(Tensor grad, Tensor softmax, u32 axis) {
  auto res = tensor_new(softmax->ndims, softmax->shape);

  auto reduce_axis = axis;
  u32 outer = 1;
  for (u32 i = 0; i < reduce_axis; ++i) {
    outer *= softmax->shape[i];
  }

  u32 inner = 1;
  for (u32 i = reduce_axis + 1; i < softmax->ndims; ++i) {
    inner *= softmax->shape[i];
  }

  auto target = softmax->shape[axis];

  if (axis + 1 == softmax->ndims) {
    dsoftmax_last_axis(outer, target, softmax->data->data, grad->data->data,
                       res->data->data);
  } else {
    dsoftmax_not_last_axis(outer, target, inner, softmax->data->data,
                           grad->data->data, res->data->data);
  }

  return res;
}

/*
Tensor tensor_softmax_all(const Tensor tensor) {
  if (!tensor)
    return nullptr;

  Tensor res = nullptr;

  Tensor max = tensor_new(1, (u32[]){1});
  if (max == nullptr)
    return nullptr;
  tensor_fill(max, tensor_max_all(tensor));

  Tensor z = tensor_sub(tensor, max);
  if (z == nullptr)
    goto free_max;

  Tensor expz = tensor_exp(z);
  if (expz == nullptr)
    goto free_z;

  Tensor sum = tensor_new(1, (u32[]){1});
  if (sum == nullptr)
    goto free_expz;
  tensor_fill(sum, tensor_sum_all(expz));

  res = tensor_div(expz, sum);

  tensor_destroy(sum);
free_expz:
  tensor_destroy(expz);
free_z:
  tensor_destroy(z);
free_max:
  tensor_destroy(max);

  return res;
}
*/
