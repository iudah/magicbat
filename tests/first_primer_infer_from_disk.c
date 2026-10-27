#include "first_primer_data.h"
#include "layer_base.h"
#include "layer_bin.h"
#include "layer_norm.h"
#include "linear_layer.h"
#include "sgd.h"
#include "tensor.h"
#include "tensor_arena.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"
#include "type_alias.h"
#include "var.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SEQ_SIZE   32
#define BATCH_SIZE 8
#define NHEAD      2
#define HEAD_DIM   8
#define D_MODEL    NHEAD *HEAD_DIM
#define LR         0.01f
#define EPOCH      400
#define GRAD_THR   1.0f
#define VAL_CLAMP  5.0f

#define kB(x) ((u64)(x * 1024))
#define MB(x) (kB(kB(x)))

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

  auto transformer_0 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  auto transformer_1 = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);

  auto layer_norm_out  = layer_norm_new(D_MODEL);
  auto lang_model_head = linear_layer_new(D_MODEL, VOCAB_SIZE);

  layers[0] = transformer_0;
  layers[1] = transformer_1;
  layers[2] = layer_norm_out;
  layers[3] = lang_model_head;
}

int main() {
  auto setup_arena = tarena_new(MB(4));
  auto train_arena = tarena_new(MB(4));
  auto infer_arena = train_arena;

  set_active_arena(setup_arena);

  auto bin       = layers_bin_open(getenv("HOME"), ".magicbat.first_primer.v1");
  mem  layers[4] = {nullptr};
  if (bin) {
    // u32 nlayers;
    // fread(&nlayers, 1, sizeof(nlayers), bin);
    layers_read(layers, 4, bin->fp);
  } else {
    abort();
  }
  Tensor weight_matrix =
      to_var(bin != nullptr ? tensor_read(bin->fp)
                            : tensor_new(2, (u32[]){VOCAB_SIZE, D_MODEL}));
  Tensor positional_encoder =
      bin != nullptr ? tensor_read(bin->fp)
                     : tensor_positional_encoding(SEQ_SIZE, D_MODEL);

  // INFERENCE / TEXT GENERATION PHASE
  printf("\n"
         "---------------------------------------------\n"
         "          Model Inference Generation         \n"
         "=============================================\n");

  set_active_arena(setup_arena);
#define INF_LIMIT (256)
  char inf_buffer[INF_LIMIT];
  u32  inf_limit = 0;

  // Tracking input tensor matching batch expectations
  Tensor inf_input = tensor_new(2, (u32[]){1, SEQ_SIZE});

  // Seed the context with space tokens (itos_map[1] is ' ')
  tensor_fill(inf_input, 1.0F);

#define LAST_TOKEN_POS (SEQ_SIZE - 1)
// Seed a prompt character into the very last position of the context
// (e.g., 'A')
#define SEED_TOKEN 23 // 'A' in your itos_map
  inf_input->data->data[LAST_TOKEN_POS] = (f32)SEED_TOKEN;

  printf("Seed Prompt: %c\nGenerated Text: ", itos_map[SEED_TOKEN]);
  fflush(stdout);

  untrack(weight_matrix);

  set_active_arena(infer_arena);
  tarena_reset(infer_arena);
#define STEP_SIZE 120
  // Autoregressively generate 120 characters
  for (u32 step = 0; step < STEP_SIZE; ++step) {
    // Forward pass without tracking dependencies for optimization gradients
    Tensor x_emb     = tensor_embedding(inf_input, weight_matrix);
    Tensor pos_input = tensor_add(x_emb, positional_encoder);

    auto logits = layers_forward(layers, 4, pos_input);

    // Argmax loop over vocabulary dimensions to extract the highest logit
    // prediction
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
      // Print out the decoded text character in real time
      printf("%s", inf_buffer);
      inf_limit     = 0;
      inf_buffer[0] = 0;
      fflush(stdout);
    } else {
      inf_buffer[inf_limit++] = itos_map[predicted_token];
      inf_buffer[inf_limit]   = 0;
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
    tensor_destroy(logits);

    tarena_reset(infer_arena);
  }

  printf("%s", inf_buffer);
  printf("\n\n");
  tensor_destroy(inf_input);

  return 0;
}
