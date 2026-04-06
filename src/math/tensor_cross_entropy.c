#include "../../include/tensor.h"
#include <stdint.h>

Tensor tensor_cross_entropy_from_logits_axis(const Tensor logits,
                                             const Tensor expectation,
                                             TensorType tensor_type, i32 axis) {
  if (!logits || !expectation)
    return NULL;

  Tensor res = NULL;

  Tensor logsumexp = tensor_log_sum_exp_axis(logits, axis);
  if (logsumexp == NULL)
    return NULL;

  Tensor loss = NULL;
  if (tensor_type & TENSOR_CE_INDEXED) {
    auto selection = tensor_gather_axis(logits, expectation, axis);
    if (selection) {
      loss = tensor_sub(logsumexp, selection);
      tensor_destroy(selection);
    }
  } else if (tensor_type & (TENSOR_CE_ONE_HOT | TENSOR_CE_PROBS)) {
    Tensor yx = tensor_mul(expectation, logits);
    if (yx) {
      auto sum_yx = tensor_sum_axis(yx, axis);
      if (sum_yx) {
        loss = tensor_sub(logsumexp, sum_yx);
      }
      tensor_destroy(yx);
    }
  }
  if (loss == NULL)
    goto free_logsumexp;

  res = loss;

free_logsumexp:
  tensor_destroy(logsumexp);

  return res;
}
