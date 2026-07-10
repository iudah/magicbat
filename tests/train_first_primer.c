#include "train_first_primer.h"
#include "layer_norm.h"
#include "linear_layer.h"
#include "sgd.h"
#include "tensor.h"
#include "transformer_enc_layer.h"
#include "var.h"
#include <stdio.h>
#include <string.h>

#define SEQ_SIZE 32
#define BATCH_SIZE 8
#define NHEAD 2
#define HEAD_DIM 8
#define D_MODEL NHEAD *HEAD_DIM
#define LR 0.01f
#define EPOCH 100

u64 urand();

void fill_input(Tensor input, Tensor target, const i32 *train_data,
                u32 train_data_length) {
  auto max_train_data_length = train_data_length - input->shape[1] - 1;

  for (u32 i = 0; i < input->shape[0]; ++i) {
    for (u32 j = 0; j < input->shape[1]; ++j) {
      u32 random_seq_indx = urand() % max_train_data_length;
      memcpy(&input->data->data[i * input->shape[1]],
             &train_data[random_seq_indx * input->shape[1]],
             sizeof(f32) * input->shape[1]);
      memcpy(&target->data->data[i * target->shape[1]],
             &train_data[(random_seq_indx * target->shape[1]) + 1],
             sizeof(f32) * target->shape[1]);
    }
  }
}

int main() {
  auto transformer_0 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  auto transformer_1 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);

  auto layer_norm_out = layer_norm_new(D_MODEL);
  auto lang_model_head = linear_layer_new(D_MODEL, D_MODEL);

  transformer_enc_layer_track(transformer_0);
  transformer_enc_layer_track(transformer_1);
  layer_norm_track(layer_norm_out);
  linear_layer_track(lang_model_head);

  Tensor kernels[(2 * N_XFORMER_KERNELS) + N_LAYERNORM_KERNELS +
                 N_LINEAR_KERNELS] = {nullptr};

  transformer_enc_layer_kernels(transformer_0, kernels, N_XFORMER_KERNELS);
  transformer_enc_layer_kernels(transformer_1, &kernels[N_XFORMER_KERNELS],
                                N_XFORMER_KERNELS);
  layer_norm_kernels(layer_norm_out,
                     kernels + N_XFORMER_KERNELS + N_XFORMER_KERNELS,
                     N_LAYERNORM_KERNELS);
  linear_layer_kernels(lang_model_head,
                       kernels + N_XFORMER_KERNELS + N_XFORMER_KERNELS +
                           N_LAYERNORM_KERNELS,
                       N_LINEAR_KERNELS);

  auto optim =
      sgd_optimizer_new(kernels, sizeof(kernels) / sizeof(*kernels), LR);

  Tensor positional_encoder = tensor_positional_encoding(SEQ_SIZE, D_MODEL);

  Tensor weight_matrix =
      track_replace_untracked(tensor_new(2, (u32[]){VOCAB_SIZE, D_MODEL}));
  Tensor input = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});
  Tensor target = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});

  for (u32 epoch = 0; epoch < EPOCH; ++epoch) {
#define TRAIN_SIZE VOCAB_SIZE * 7 / 10
    fill_input(input, target, (i32 *)train_data, TRAIN_SIZE);
    Tensor x_emb = var_embedding(input, weight_matrix);
    Tensor y_emb = var_embedding(target, weight_matrix);

    auto block_out = transformer_enc_layer_forward(
        transformer_0, var_add(x_emb, positional_encoder));
    auto model_out = transformer_enc_layer_forward(transformer_1, block_out);

    auto final_norm = layer_norm_forward(layer_norm_out, model_out);
    auto logits = linear_layer_forward(lang_model_head, final_norm);

    auto loss = var_cross_entropy_probs(logits, y_emb, -1);

    printf("Epoch %d | Loss: %f\n", epoch, loss->data->data[0]);

    var_backward(loss);
    sgd_optimize(optim, nullptr);
  }
}
