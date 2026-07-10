#include "test_crossentropy.h"
#include "tensor.h"
#include "var.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define EPS 1e-4

bool verify_tensor(Tensor out, f32 *target, const char *test_name) {
  if (!out) {
    printf("[FAIL] %s: Output is null.\n", test_name);
    return false;
  }
  for (u32 i = 0; i < out->data->nelements; ++i) {
    if (fabsf(out->data->data[i] - target[i]) > EPS) {
      printf("[FAIL] %s: Mismatch at index %u (Expected: %f, Got: %f)\n",
             test_name, i, target[i], out->data->data[i]);
      return false;
    }
  }
  printf("[PASS] %s\n", test_name);
  return true;
}

// Assumes definitions for var_crossentropy_indexed, var_crossentropy_one_hot,
// var_crossentropy_probs
int main() {
  u32 shape2D[] = {BATCH_SIZE, VOCAB_SIZE};
  u32 shape1D[] = {BATCH_SIZE};

  auto dLoss = tensor_new(1, shape1D);
  memcpy(dLoss->data->data, dLoss_np, BATCH_SIZE * sizeof(f32));

  // ---------------------------------------------------------
  // 1. ONE-HOT PATH
  // ---------------------------------------------------------
  auto logits_oh = track_replace_untracked(tensor_new(2, shape2D));
  memcpy(logits_oh->data->data, logits_np,
         BATCH_SIZE * VOCAB_SIZE * sizeof(f32));

  auto targets_oh = tensor_new(2, shape2D);
  memcpy(targets_oh->data->data, targets_one_hot_np,
         BATCH_SIZE * VOCAB_SIZE * sizeof(f32));

  auto loss_oh = var_cross_entropy_one_hot(logits_oh, targets_oh, 1);
  verify_tensor(loss_oh, loss_one_hot_target, "One-Hot Forward");

  var_backward_with_grad(loss_oh, dLoss);
  verify_tensor(var_grad(logits_oh), dX_one_hot_target, "One-Hot Backward");

  // ---------------------------------------------------------
  // 2. PROBS PATH
  // ---------------------------------------------------------
  auto logits_pr = track_replace_untracked(tensor_new(2, shape2D));
  memcpy(logits_pr->data->data, logits_np,
         BATCH_SIZE * VOCAB_SIZE * sizeof(f32));

  auto targets_pr = tensor_new(2, shape2D);
  memcpy(targets_pr->data->data, targets_probs_np,
         BATCH_SIZE * VOCAB_SIZE * sizeof(f32));

  auto loss_pr = var_cross_entropy_probs(logits_pr, targets_pr, 1);
  verify_tensor(loss_pr, loss_probs_target, "Probs Forward");

  var_backward_with_grad(loss_pr, dLoss);
  verify_tensor(var_grad(logits_pr), dX_probs_target, "Probs Backward");

  // ---------------------------------------------------------
  // 3. INDEXED PATH
  // ---------------------------------------------------------
  auto logits_idx = track_replace_untracked(tensor_new(2, shape2D));
  memcpy(logits_idx->data->data, logits_np,
         BATCH_SIZE * VOCAB_SIZE * sizeof(f32));

  auto targets_idx = tensor_new(1, shape1D);
  for (u32 i = 0; i < BATCH_SIZE; ++i) {
    targets_idx->data->data[i] = (f32)
        targets_indexed_np[i]; // Store as float if tensor structure requires
                               // it, or handle integer tensors accordingly
  }

  auto loss_idx = var_cross_entropy_indexed(logits_idx, targets_idx, 1);
  verify_tensor(loss_idx, loss_indexed_target, "Indexed Forward");

#ifndef UNDEF_WHEN_INDEXED_BP_IS_IMPLEMENTED
  printf("[SKIP] Indexed Backward (Not implemented yet)\n");
#else
  var_backward_with_grad(loss_idx, dLoss);
  verify_tensor(var_grad(logits_idx), dX_indexed_target, "Indexed Backward");
#endif

  return 0;
}
