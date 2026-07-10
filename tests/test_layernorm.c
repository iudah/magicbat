#include "test_layernorm.h"
#include "tensor.h"
#include "var.h"
#include <math.h>
#include <stdio.h>

#define BATCH_SIZE 2
#define SEQ_LEN 8
#define NHEADS 2
#define HEAD_DIM 8
#define D_MODEL HEAD_DIM *NHEADS

int main() {

  // Use IDENTICAL starting values as the discrete test
  auto input = track_replace_untracked(
      tensor_new(3, (u32[]){BATCH_SIZE, SEQ_LEN, D_MODEL}));
  auto grad = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_LEN, D_MODEL});

  for (u32 i = 0; i < input->data->nelements; ++i) {
    input->data->data[i] = X[i];
  }
  for (u32 i = 0; i < grad->data->nelements; ++i) {
    grad->data->data[i] = dy[i];
  }

  // 1. Forward Pass
  auto out = var_layer_norm_axis(input, 2);

  // 2. Backward
  var_backward_with_grad(out, grad);

  auto dinput = var_grad(input);

#define EPS 1e-4

  bool success = true;
  for (u32 i = 0; i < out->data->nelements; ++i) {

    if (fabsf(out->data->data[i] - out_target[i]) > EPS) {
      success = false;
      break;
    }
  }
  printf("Test %s on layernorm.\n", success ? "succeed" : "failed");

  success = true;
  for (u32 i = 0; i < dinput->data->nelements; ++i) {

    if (fabsf(dinput->data->data[i] - dX_target[i]) > EPS) {
      success = false;
      break;
    }
  }
  printf("Test %s on layernorm gradient.\n", success ? "succeed" : "failed");
}
