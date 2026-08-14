#include "layer_norm.h"
#include "linear_layer.h"
#include "sgd.h"
#include "tensor.h"
#include "train_first_primer.h"
#include "transformer_enc_layer.h"
#include "var.h"
#include <math.h>
#include <stdio.h>

#define SEQ_SIZE 32
#define BATCH_SIZE 8
#define NHEAD 2
#define HEAD_DIM 8
#define D_MODEL NHEAD *HEAD_DIM
#define LR 0.01f
#define EPOCH 300

u64 urand();

void fill_input(Tensor input, Tensor target, const i32 *train_data,
                u32 train_data_length) {
  auto max_train_data_length = train_data_length - input->shape[1] - 1;

  for (u32 i = 0; i < input->shape[0]; ++i) {
    u32 random_seq_indx = urand() % max_train_data_length;
    for (u32 j = 0; j < input->shape[1]; ++j) {
      input->data->data[(i * input->shape[1]) + j] =
          train_data[random_seq_indx + j];
      target->data->data[(i * input->shape[1]) + j] =
          train_data[random_seq_indx + j + 1];
    }
  }
}

int main() {
  auto transformer_0 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  auto transformer_1 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);

  auto layer_norm_out = layer_norm_new(D_MODEL);
  auto lang_model_head = linear_layer_new(D_MODEL, VOCAB_SIZE);

  transformer_enc_layer_track(transformer_0);
  transformer_enc_layer_track(transformer_1);
  layer_norm_track(layer_norm_out);
  linear_layer_track(lang_model_head);

  u32 total_kernels =
      (2 * N_XFORMER_KERNELS) + N_LAYERNORM_KERNELS + N_LINEAR_KERNELS;
  Tensor kernels[(2 * N_XFORMER_KERNELS) + N_LAYERNORM_KERNELS +
                 N_LINEAR_KERNELS + 1] = {nullptr};

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
  Tensor weight_matrix =
      track_replace_untracked(tensor_new(2, (u32[]){VOCAB_SIZE, D_MODEL}));
  tensor_random_bound(weight_matrix, 0, 1);
  kernels[total_kernels - 1] = weight_matrix;

  auto optim = sgd_optimizer_new(kernels, total_kernels, LR);

  Tensor positional_encoder = tensor_positional_encoding(SEQ_SIZE, D_MODEL);

  Tensor input = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});
  Tensor target = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});

  // TRAINING PHASE
  for (u32 epoch = 0; epoch < EPOCH; ++epoch) {
#define TRAIN_SIZE VOCAB_SIZE * 7 / 10
    fill_input(input, target, (i32 *)train_data, TRAIN_SIZE);
    Tensor x_emb = var_embedding(input, weight_matrix);

    auto block_out = transformer_enc_layer_forward(
        transformer_0, var_add(x_emb, positional_encoder));
    auto model_out = transformer_enc_layer_forward(transformer_1, block_out);

    auto final_norm = layer_norm_forward(layer_norm_out, model_out);
    auto logits = linear_layer_forward(lang_model_head, final_norm);

    auto loss = var_cross_entropy_loss_indexed(logits, target, -1);

#define EPOCH_CHECKPOINT 50
    if (epoch % EPOCH_CHECKPOINT == 0)
      printf("Epoch %d | Loss: %f\n", epoch, loss->data->data[0]);

    var_backward(loss);
    sgd_optimize(optim, nullptr);

    for (u32 k = 0; k < total_kernels; ++k) {
      if (kernels[k]) {
        auto grad = var_grad(kernels[k]);
        tensor_fill(grad, 0.0F);
      }
    }
  }

  // INFERENCE / TEXT GENERATION PHASE
  printf("\n"
         "---------------------------------------------\n"
         "          Model Inference Generation         \n"
         "=============================================\n");

#define INF_LIMIT (256)
  char inf_buffer[INF_LIMIT];
  u32 inf_limit = 0;

  // Tracking input tensor matching batch expectations
  Tensor inf_input = tensor_new(2, (u32[]){1, SEQ_SIZE});

  // Seed the context with space tokens (itos_map[1] is ' ')
  tensor_fill(inf_input, 1.0F);

#define LAST_TOKEN_POS (SEQ_SIZE - 1)
// Seed a prompt character into the very last position of the context
// (e.g., 'A')
#define SEED_TOKEN 23 // 'A' in your itos_map
  for (u32 batch = 0; batch < BATCH_SIZE; ++batch) {
    inf_input->data->data[(batch * SEQ_SIZE) + LAST_TOKEN_POS] =
        (f32)SEED_TOKEN;
  }

  printf("Seed Prompt: %c\nGenerated Text: ", itos_map[SEED_TOKEN]);
  fflush(stdout);

#define STEP_SIZE 120
  // Autoregressively generate 120 characters
  for (u32 step = 0; step < STEP_SIZE; ++step) {
    // Forward pass without tracking dependencies for optimization gradients
    Tensor x_emb = tensor_embedding(inf_input, weight_matrix);
    Tensor pos_input = tensor_add(x_emb, positional_encoder);

    Tensor b0_out = transformer_enc_layer_forward(transformer_0, pos_input);
    Tensor b1_out = transformer_enc_layer_forward(transformer_1, b0_out);

    Tensor f_norm = layer_norm_forward(layer_norm_out, b1_out);
    Tensor logits = linear_layer_forward(lang_model_head, f_norm);

    // Argmax loop over vocabulary dimensions to extract the highest logit
    // prediction
    u32 predicted_token = 0;
    f32 max_score = -INFINITY;
    for (u32 vocab = 0; vocab < VOCAB_SIZE; ++vocab) {
      f32 score =
          logits->data
              ->data[(((0 * SEQ_SIZE) + LAST_TOKEN_POS) * VOCAB_SIZE) + vocab];
      if (score > max_score) {
        max_score = score;
        predicted_token = vocab;
      }
    }

    if (inf_limit >= (INF_LIMIT - 1)) {
      // Print out the decoded text character in real time
      printf("%s", inf_buffer);
      inf_limit = 0;
      inf_buffer[0] = 0;
      fflush(stdout);
    } else {
      inf_buffer[inf_limit++] = itos_map[predicted_token];
      inf_buffer[inf_limit] = 0;
    }

    // Roll context window left by 1 element across the batch to make room for
    // autoregression step
    for (u32 j = 0; j < SEQ_SIZE - 1; ++j) {
      inf_input->data->data[j] = inf_input->data->data[j + 1];
    }
    // Insert the freshly predicted character ID to the end of the sliding
    // window context
    inf_input->data->data[LAST_TOKEN_POS] = (f32)predicted_token;

    // Clean up inference epoch tensor nodes
    tensor_destroy(x_emb);
    tensor_destroy(pos_input);
    tensor_destroy(b0_out);
    tensor_destroy(b1_out);
    tensor_destroy(f_norm);
    tensor_destroy(logits);
  }

  printf("%s", inf_buffer);
  printf("\n\n");
  tensor_destroy(inf_input);
  return 0;
}
