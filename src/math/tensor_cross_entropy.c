#include "tensor.h"
#include "tensor_binary_op.h"
#include "tensor_memory.h"
#include "tensor_trinary_op.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

Tensor tensor_cross_entropy_from_logits_axis(const Tensor logits,
                                             const Tensor expectation,
                                             TensorType tensor_type, i32 axis) {
  if (!logits || !expectation)
    return nullptr;

  Tensor res = nullptr;

  Tensor logsumexp = tensor_log_sum_exp_axis(logits, axis);
  if (logsumexp == nullptr)
    return nullptr;

  Tensor loss = nullptr;
  if (tensor_type & TENSOR_CE_INDEXED) {
    auto selection = tensor_gather_axis(logits, expectation, axis);
    if (selection) {
      loss = tensor_sub(logsumexp, selection);
      tensor_destroy(selection);
    }
  } else if (tensor_type & (TENSOR_CE_ONE_HOT | TENSOR_CE_PROBS)) {
    Tensor exp_logits = tensor_mul(expectation, logits);
    if (exp_logits) {
      auto sum_yx = tensor_sum_axis(exp_logits, axis);
      if (sum_yx) {
        loss = tensor_sub(logsumexp, sum_yx);
      }
      tensor_destroy(exp_logits);
    }
  }
  if (loss == nullptr)
    goto free_logsumexp;

  res = loss;

free_logsumexp:
  tensor_destroy(logsumexp);

  return res;
}

Tensor tensor_ce_backward(Tensor logits, Tensor expectation, Tensor grad,
                          i32 axis);
Tensor tensor_ce_contiguous_backward(Tensor logits, Tensor expectation,
                                     Tensor grad, i32 axis);
f32 ce_add_mul(f32 probs, f32 expectation, f32 grad) {
  return (probs - expectation) * grad;
}
Tensor tensor_ce_add_mul(Tensor probs, Tensor expectation, Tensor grad) {
  if (grad)
    return tensor_trinary_op(probs, expectation, grad, ce_add_mul);
  return tensor_binary_op(probs, expectation, 1, ce_add_mul);
}

Tensor tensor_cross_entropy_from_logits_axis_backward(const Tensor logits,
                                                      const Tensor expectation,
                                                      const Tensor grad,
                                                      TensorType tensor_type,
                                                      i32 axis) {
  if (!logits->is_contiguous || !expectation->is_contiguous ||
      !grad->is_contiguous) {
    if (tensor_type != TENSOR_CE_INDEXED)
      return tensor_ce_backward(logits, expectation, grad, axis);
    printf("Backprop on cross entropy for TENSOR_CE_INDEXED not supported.");
    abort();
  }

  if (tensor_type != TENSOR_CE_INDEXED)
    return tensor_ce_contiguous_backward(logits, expectation, grad, axis);

  printf("Backprop on cross entropy for TENSOR_CE_INDEXED not supported.");
  return nullptr;
}

Tensor tensor_ce_backward(const Tensor logits, const Tensor expectation,
                          const Tensor grad, i32 axis) {
  auto probs = tensor_softmax(logits, axis);
  auto res = tensor_ce_add_mul(probs, expectation, grad);
  tensor_destroy(probs);
  return res;
}
static void softmax_last_axis(u32 outer, u32 target, const f32 *restrict logits,
                              const f32 *restrict expectation,
                              const f32 *restrict grad, f32 *restrict output);
static void softmax_not_last_axis(u32 outer, u32 target, u32 inner,
                                  const f32 *restrict logits,
                                  const f32 *restrict expectation,
                                  const f32 *restrict grad,
                                  f32 *restrict output);
static void softmax_last_axis_null_grad(u32 outer, u32 target,
                                        const f32 *restrict logits,
                                        const f32 *restrict expectation,
                                        f32 *restrict output);
static void softmax_not_last_axis_null_grad(u32 outer, u32 target, u32 inner,
                                            const f32 *restrict logits,
                                            const f32 *restrict expectation,
                                            f32 *restrict output);
Tensor tensor_ce_contiguous_backward(const Tensor logits,
                                     const Tensor expectation,
                                     const Tensor grad, i32 axis) {
  if (!logits)
    return nullptr;

  auto res = tensor_new(logits->ndims, logits->shape);

  u32 reduce_axis = axis < 0 ? logits->ndims + axis : axis;
  if (!logits->is_contiguous) {
    abort();
    return nullptr;
  }

  u32 outer = 1;
  for (u32 i = 0; i < reduce_axis; ++i) {
    outer *= logits->shape[i];
  }

  u32 inner = 1;
  for (u32 i = reduce_axis + 1; i < logits->ndims; ++i) {
    inner *= logits->shape[i];
  }

  if (reduce_axis + 1 == logits->ndims && grad) {
    softmax_last_axis(outer, logits->shape[reduce_axis], logits->data->data,
                      expectation->data->data, grad->data->data,
                      res->data->data);
  } else if (reduce_axis + 1 != logits->ndims && grad) {
    softmax_not_last_axis(outer, logits->shape[reduce_axis], inner,
                          logits->data->data, expectation->data->data,
                          grad->data->data, res->data->data);
  } else if (reduce_axis + 1 == logits->ndims && !grad) {
    softmax_last_axis_null_grad(outer, logits->shape[reduce_axis],
                                logits->data->data, expectation->data->data,
                                res->data->data);
  } else if (reduce_axis + 1 != logits->ndims && !grad) {
    softmax_not_last_axis_null_grad(outer, logits->shape[reduce_axis], inner,
                                    logits->data->data, expectation->data->data,
                                    res->data->data);
  }

  return res;
}
#define EPS (1e-6)
static void softmax_last_axis(u32 outer, u32 target, const f32 *restrict logits,
                              const f32 *restrict expectation,
                              const f32 *restrict grad, f32 *restrict output) {
  for (u32 i = 0; i < outer; ++i) {
    f32 sum = 0;
    f32 max = -INFINITY;
    for (u32 j = 0; j < target; ++j) {
      f32 score = logits[(i * target) + j];
      f32 new_max = fmaxf(max, score);
      f32 probability = expf(score - new_max);
      f32 diff = max - new_max;
      f32 alpha = expf(diff);
      sum = (sum * alpha) + probability;
      max = new_max;
    }
    for (u32 j = 0; j < target; ++j) {
      f32 score = logits[(i * target) + j];
      f32 probability = expf(score - max);
      output[(i * target) + j] =
          ((probability / sum) - expectation[(i * target) + j]) * grad[i];
    }
  }
}
static void softmax_not_last_axis(u32 outer, u32 target, u32 inner,
                                  const f32 *restrict logits,
                                  const f32 *restrict expectation,
                                  const f32 *restrict grad,
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
        f32 score = logits[((i * target + j) * inner) + k];
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
        f32 score = logits[((i * target + j) * inner) + k];
        f32 max = max_array[k];
        f32 sum = sum_array[k];
        f32 probability = expf(score - max);
        output[((i * target + j) * inner) + k] =
            ((probability / sum) -
             expectation[((i * target + j) * inner) + k]) *
            grad[(i * inner) + k];
      }
    }
  }
  tfree(sum_array);
  tfree(max_array);
}
static void softmax_last_axis_null_grad(u32 outer, u32 target,
                                        const f32 *restrict logits,
                                        const f32 *restrict expectation,
                                        f32 *restrict output) {
  for (u32 i = 0; i < outer; ++i) {
    f32 sum = 0;
    f32 max = -INFINITY;
    for (u32 j = 0; j < target; ++j) {
      f32 score = logits[(i * target) + j];
      f32 new_max = fmaxf(max, score);
      f32 probability = expf(score - new_max);
      f32 diff = max - new_max;
      f32 alpha = expf(diff);
      sum = (sum * alpha) + probability;
      max = new_max;
    }
    for (u32 j = 0; j < target; ++j) {
      f32 score = logits[(i * target) + j];
      f32 probability = expf(score - max);
      output[(i * target) + j] =
          ((probability / sum) - expectation[(i * target) + j]);
    }
  }
}
static void softmax_not_last_axis_null_grad(u32 outer, u32 target, u32 inner,
                                            const f32 *restrict logits,
                                            const f32 *restrict expectation,
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
        f32 score = logits[((i * target + j) * inner) + k];
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
        f32 score = logits[((i * target + j) * inner) + k];
        f32 max = max_array[k];
        f32 sum = sum_array[k];
        f32 probability = expf(score - max);
        output[((i * target + j) * inner) + k] =
            ((probability / sum) - expectation[((i * target + j) * inner) + k]);
      }
    }
  }

  tfree(sum_array);
  tfree(max_array);
}
