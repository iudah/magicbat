#include "adam.h"
#include "dropout/dropout_layer.h"
#include "first_primer_data.h"
#include "layer_base.h"
#include "layer_bin.h"
#include "layer_norm.h"
#include "linear_layer.h"
#include "tensor.h"
#include "tensor_arena.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"
#include "type_alias.h"
#include "var.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SEQ_SIZE     32
#define BATCH_SIZE   32
#define NHEAD        2
#define HEAD_DIM     8
#define D_MODEL      NHEAD *HEAD_DIM
#define LR           3e-4f
#define EPOCH        400
#define GRAD_THR     1.0f
#define VAL_CLAMP    5.0f
#define BETA1        0.9f
#define BETA2        0.999f
#define EPSILON      1e-8f
#define WEIGHT_DECAY 0.1f

#define kB(x) ((u64)(x * 1024))
#define MB(x) (kB(kB(x)))

#define CKPT_NAME ".magicbat.first_primer.v1"
#define CKPT_VER  1

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

void create_model(mem layers[]) {
  auto drop_0        = dropout_layer_new(0.1);
  auto transformer_0 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  auto drop_1        = dropout_layer_new(0.1);
  auto transformer_1 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  auto drop_2 = dropout_layer_new(0.1);

  auto layer_norm_out  = layer_norm_new(D_MODEL);
  auto drop_3          = dropout_layer_new(0.1);
  auto lang_model_head = linear_layer_new(D_MODEL, VOCAB_SIZE);

  layers[0] = drop_0;
  layers[1] = transformer_0;
  layers[2] = drop_1;
  layers[3] = transformer_1;
  layers[4] = drop_2;
  layers[5] = layer_norm_out;
  layers[6] = drop_3;
  layers[7] = lang_model_head;
}

int main() {
  auto setup_arena = tarena_new(MB(4));
  auto train_arena = tarena_new(MB(400));
  auto infer_arena = train_arena;

  set_active_arena(setup_arena);

  const char *home      = getenv("HOME");
  const char *model_bin = getenv("MBAT_FILE");

  mem    layers[8]          = {nullptr};
  Tensor weight_matrix      = nullptr;
  Tensor positional_encoder = nullptr;
  u32    adam_steps         = 0;

  auto bin = layers_bin_open(home, model_bin ? model_bin : CKPT_NAME);
  if (bin) {
    layers_read(layers, 8, bin->fp);
    weight_matrix      = tensor_read(bin->fp);
    positional_encoder = tensor_read(bin->fp);
    fread(&adam_steps, sizeof(adam_steps), 1, bin->fp);
    layers_bin_close(bin);
    bin = nullptr;
  } else {
    create_model(layers);
    weight_matrix = tensor_new(2, (u32[]){VOCAB_SIZE, D_MODEL});
    tensor_random_bound(weight_matrix, 0, 1);
    positional_encoder = tensor_positional_encoding(SEQ_SIZE, D_MODEL);
    adam_steps         = 0;
  }

  layers_track(layers, 8);

#define total_kernels                                                          \
  ((2 * N_XFORMER_KERNELS) + N_LAYERNORM_KERNELS + N_LINEAR_KERNELS + 1)
  Tensor kernels[total_kernels] = {nullptr};
  layers_collect_kernels(layers, 8, kernels, total_kernels);

  weight_matrix              = to_var(weight_matrix);
  kernels[total_kernels - 1] = weight_matrix;

  auto optim =
      adamw_optimizer_new(kernels, total_kernels, LR, GRAD_THR, VAL_CLAMP,
                          BETA1, BETA2, WEIGHT_DECAY, (f32)adam_steps, EPSILON);

  Tensor input  = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});
  Tensor target = tensor_new(2, (u32[]){BATCH_SIZE, SEQ_SIZE});

  // TRAINING PHASE
  set_active_arena(train_arena);
  for (u32 epoch = 0; epoch < EPOCH; ++epoch) {
#define TRAIN_SIZE (DATA_LEN * 7 / 10)
    fill_input(input, target, (i32 *)train_data, TRAIN_SIZE);
    Tensor x_emb = var_embedding(input, weight_matrix);

    auto logits = layers_forward(layers, 8, var_add(x_emb, positional_encoder));
    auto loss   = var_cross_entropy_loss_indexed(logits, target, -1);

#define EPOCH_CHECKPOINT 50
    if ((epoch + 1) % EPOCH_CHECKPOINT == 0 || epoch == (EPOCH - 1) ||
        epoch == 0)
      printf("Epoch %d | Loss: %f\n", epoch + 1, loss->data->data[0]);

#define EPOCH_SAVEPOINT 100
    if (epoch != 0 &&
        ((epoch + 1) % EPOCH_SAVEPOINT == 0 || epoch == (EPOCH - 1))) {
      printf("Epoch %d | Serialized\n", epoch + 1);
      bin =
          layers_bin_create(home, model_bin ? model_bin : CKPT_NAME, CKPT_VER);
      if (!bin) {
        printf("failed to create model binary");
        abort();
      }
      if (!layers_write(layers, 8, bin->fp)) {
        printf("Failed to save layers");
        abort();
      };
      tensor_write(weight_matrix, bin->fp);
      tensor_write(positional_encoder, bin->fp);
      fwrite(&adam_steps, sizeof(adam_steps), 1, bin->fp);
      layers_bin_close(bin);
      bin = nullptr;
    }

    var_backward(loss);
    adamw_optimize(optim, nullptr);
    ++adam_steps;

    // for (u32 k = 0; k < total_kernels; ++k) {
    //   if (kernels[k]) {
    //     auto grad = var_grad(kernels[k]);
    //     tensor_fill(grad, 0.0F);
    //   }
    // }
    var_null_grads(kernels, total_kernels);
    var_destroy(loss);
    tarena_reset(active_arena);
  }

  // INFERENCE / TEXT GENERATION PHASE
  printf("\n"
         "---------------------------------------------\n"
         "          Model Inference Generation         \n"
         "=============================================\n");

  set_active_arena(setup_arena);
#define INF_LIMIT (256)
  char inf_buffer[INF_LIMIT];
  u32  inf_limit = 0;

  Tensor inf_input = tensor_new(2, (u32[]){1, SEQ_SIZE});

  // Seed with spaces
  tensor_fill(inf_input, 1.0F);

#define LAST_TOKEN_POS (SEQ_SIZE - 1)
#define SEED_TOKEN     23 // 'A'
  inf_input->data->data[LAST_TOKEN_POS] = (f32)SEED_TOKEN;

  printf("Seed Prompt: %c\nGenerated Text: '", itos_map[SEED_TOKEN]);
  fflush(stdout);

  untrack(weight_matrix);
  layers_untrack(layers, 8);

  set_active_arena(infer_arena);
  tarena_reset(infer_arena);
#define STEP_SIZE 120
  for (u32 step = 0; step < STEP_SIZE; ++step) {
    Tensor x_emb     = tensor_embedding(inf_input, weight_matrix);
    Tensor pos_input = tensor_add(x_emb, positional_encoder);

    auto logits = layers_forward(layers, 8, pos_input);

    u32 predicted_token = 0;
    f32 max_score       = -INFINITY;
    for (u32 vocab = 0; vocab < VOCAB_SIZE; ++vocab) {
      f32 score =
          logits->data
              ->data[(((0 * SEQ_SIZE) + LAST_TOKEN_POS) * VOCAB_SIZE) + vocab];
      if (score > max_score) {
        max_score       = score;
        predicted_token = vocab;
      }
    }

    if (inf_limit >= (INF_LIMIT - 1)) {
      printf("%s", inf_buffer);
      inf_limit     = 0;
      inf_buffer[0] = 0;
      fflush(stdout);
    } else {
      inf_buffer[inf_limit++] = itos_map[predicted_token];
      inf_buffer[inf_limit]   = 0;
    }

    for (u32 j = 0; j < SEQ_SIZE - 1; ++j) {
      inf_input->data->data[j] = inf_input->data->data[j + 1];
    }
    inf_input->data->data[LAST_TOKEN_POS] = (f32)predicted_token;

    tensor_destroy(x_emb);
    tensor_destroy(pos_input);
    tensor_destroy(logits);

    tarena_reset(infer_arena);
  }

  printf("%s'", inf_buffer);
  printf("\n\n");
  tensor_destroy(inf_input);

  adam_optimizer_destroy(optim);

  return 0;
}
