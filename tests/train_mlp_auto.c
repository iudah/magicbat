#include "../include/adt/tensor/tensor_prot.h"
#include "../include/layers/linear_layer.h"
#include "../include/optimizers/sgd.h"
#include "../include/tensor.h"
#include "../include/var/var.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  printf("Autograd MLP training example...\n");

  LinearLayer fc1 = linear_layer_new(2, 4);
  LinearLayer fc2 = linear_layer_new(4, 1);

  linear_layer_track(fc1);
  linear_layer_track(fc2);

  Tensor w1 = linear_layer_weight(fc1);
  Tensor w2 = linear_layer_weight(fc2);
  Tensor b1 = linear_layer_bias(fc1);
  Tensor b2 = linear_layer_bias(fc2);

  tensor_fill(w1, 0.25f);
  tensor_fill(w2, 0.35f);
  tensor_fill(b1, 0.20f);
  tensor_fill(b2, 0.30f);

  Tensor *parameters = (Tensor[]){w1, b1, w2, b2};

  SgdOptimizer optimizer = sgd_optimizer_new(parameters, 4, 0.1f);

  Tensor x = tensor_new(2, (u32[]){4, 2});
  Tensor y = tensor_new(2, (u32[]){4, 1});

  x->data->data[0] = 0;
  x->data->data[1] = 0;
  y->data->data[0] = 0;
  x->data->data[2] = 0;
  x->data->data[3] = 1;
  y->data->data[1] = 1;
  x->data->data[4] = 1;
  x->data->data[5] = 0;
  y->data->data[2] = 1;
  x->data->data[6] = 1;
  x->data->data[7] = 1;
  y->data->data[3] = 0;

  for (int epoch = 0; epoch < 100; ++epoch) {

    Tensor h1 = linear_layer_forward(fc1, x);
    Tensor h1_act = var_relu(h1);
    Tensor logits = linear_layer_forward(fc2, h1_act);

    // MSE Loss
    Tensor diff = var_sub(logits, y);
    Tensor loss_sq = var_mul(diff, diff);

    // Loss = sum(diff^2) / N
    Tensor n_scalar =
        tensor_scalar((float)tensor_num_elements((Tensor)loss_sq));
    Tensor loss = var_div((Tensor)var_sum_all(loss_sq), n_scalar);

    printf("Epoch %d, Loss = %.4f\n", epoch, ((Tensor)loss)->data->data[0]);

    var_zero_grad(loss);

    var_backward(loss);

    sgd_optimize(optimizer, (Tensor[]){var_grad(w1), var_grad(b1), var_grad(w2),
                                       var_grad(b2)});

    var_destroy(loss);
    var_destroy(n_scalar);
  }

  sgd_optimizer_destroy(optimizer);
  linear_layer_destroy(fc2);
  linear_layer_destroy(fc1);
  tensor_destroy(y);
  tensor_destroy(x);

  printf("Autograd training example finished.\n");
  return 0;
}
