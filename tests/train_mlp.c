// #include "../include/adt/layers/layers_prot.h"
#include "../include/adt/tensor/tensor_prot.h"
#include "../include/layers/linear_layer.h"
#include "../include/optimizers/sgd.h"
#include "../include/tensor.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  printf("Simple MLP training example...\n");

  // Create a tiny 2-layer MLP: 2 -> 4 -> 1
  LinearLayer fc1 = linear_layer_new(2, 4);
  LinearLayer fc2 = linear_layer_new(4, 1);

  Tensor *parameters =
      (Tensor[]){linear_layer_weight(fc1), linear_layer_bias(fc1),
                 linear_layer_weight(fc2), linear_layer_bias(fc2)};

  SgdOptimizer optimizer = sgd_optimizer_new(parameters, 4, 0.1f);

  // Tiny dataset: 4 samples, 2 features
  Tensor x = tensor_new(2, (u32[]){4, 2});
  Tensor y = tensor_new(2, (u32[]){4, 1});

  // Dummy data (XOR-like)
  x->data[0] = 0;
  x->data[1] = 0;
  y->data[0] = 0;
  x->data[2] = 0;
  x->data[3] = 1;
  y->data[1] = 1;
  x->data[4] = 1;
  x->data[5] = 0;
  y->data[2] = 1;
  x->data[6] = 1;
  x->data[7] = 1;
  y->data[3] = 0;

  tensor_fill(linear_layer_weight(fc1), 0.25f);
  tensor_fill(linear_layer_weight(fc2), 0.35f);
  tensor_fill(linear_layer_bias(fc1), 0.20f);
  tensor_fill(linear_layer_bias(fc2), 0.30f);

  for (int epoch = 0; epoch < 100; ++epoch) {
    // Forward pass
    Tensor h1 = linear_layer_forward(fc1, x);
    Tensor h1_act = tensor_relu(h1);
    Tensor logits = linear_layer_forward(fc2, h1_act);

    // Loss (MSE for simplicity)
    Tensor diff = tensor_sub(logits, y);
    Tensor loss_sq = tensor_mul(diff, diff);
    Tensor loss =
        tensor_scalar(tensor_sum_all(loss_sq) / tensor_num_elements(loss_sq));

    printf("Epoch %d, Loss = %.4f\n", epoch, loss->data[0]);

    // === Manual Backprop (simple chain rule) ===

    // dLoss/dLogits = 2 * (logits - y) / N
    Tensor two = tensor_scalar(2.0f);
    Tensor grad_logits =
        tensor_div(tensor_mul(two, diff),
                   tensor_scalar((float)tensor_num_elements(logits)));

    // Layer 2 backprop
    Tensor grad_w2 = tensor_matmul(tensor_transpose(h1_act), grad_logits);
    Tensor grad_b2 = tensor_sum_axis(grad_logits, 0);

    // Layer 1 backprop (through ReLU)
    Tensor grad_h1_act =
        tensor_matmul(grad_logits, tensor_transpose(linear_layer_weight(fc2)));
    Tensor grad_h1 = tensor_relu_backward(h1, grad_h1_act);

    Tensor grad_w1 = tensor_matmul(tensor_transpose(x), grad_h1);
    Tensor grad_b1 = tensor_sum_axis(grad_h1, 0);

    // Update using SGD
    sgd_optimize(optimizer, (Tensor[]){grad_w2, grad_b2, grad_w1, grad_b1});

    // Cleanup
    tensor_destroy(grad_logits);
    tensor_destroy(grad_w2);
    tensor_destroy(grad_b2);
    tensor_destroy(grad_h1_act);
    tensor_destroy(grad_h1);
    tensor_destroy(loss);
    tensor_destroy(loss_sq);
    tensor_destroy(diff);
    tensor_destroy(h1_act);
    tensor_destroy(h1);
    tensor_destroy(logits);
  }

  // Cleanup
  sgd_optimizer_destroy(optimizer);
  linear_layer_destroy(fc2);
  linear_layer_destroy(fc1);
  tensor_destroy(y);
  tensor_destroy(x);

  printf("Training example finished.\n");
  return 0;
}