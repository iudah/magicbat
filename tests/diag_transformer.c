// tests/diagnose_transformer.c
//
// Diagnostic harness for the training plateau.  Run and read the output;
// no training is required beyond TEST 4's tiny head-to-head.

#include "layer_norm.h"
#include "layers_prot.h"
#include "linear_layer.h"
#include "sgd.h"
#include "tensor.h"
#include "tensor_arena.h"
#include "tensor_memory.h"
#include "transformer_enc_layer.h"
#include "var.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define kB(x) ((u64)(x * 1024))
#define MB(x) (kB(kB(x)))

#define SEQ_SIZE   32
#define BATCH_SIZE 8
#define NHEAD      2
#define HEAD_DIM   8
#define D_MODEL    (NHEAD * HEAD_DIM)
#define LR         3e-3f
#define GRAD_THR   1.0f
#define VAL_CLAMP  5.0f

/* ------------------------------------------------------------------ */
/* residual strategies we want to compare against the codebase's      */
/* ------------------------------------------------------------------ */

/* Standard pre-LN:   x + Sublayer( LN(x) )  */
static Tensor std_pre_ln(LayerNorm ln, Tensor input, mem sublayer,
                         Tensor (*sublayer_forward)(mem, Tensor)) {
  return var_add(input,
                 sublayer_forward(sublayer, layer_norm_forward(ln, input)));
}

/* Standard post-LN:  LN( x + Sublayer(x) )  */
static Tensor std_post_ln(LayerNorm ln, Tensor input, mem sublayer,
                          Tensor (*sublayer_forward)(mem, Tensor)) {
  return layer_norm_forward(
      ln, var_add(input, sublayer_forward(sublayer, input)));
}

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static f32 tensor_variance(Tensor t) {
  u32 n   = tensor_num_elements(t);
  f32 mean = 0;
  for (u32 i = 0; i < n; ++i) mean += t->data->data[i];
  mean /= (f32)n;
  f32 var = 0;
  for (u32 i = 0; i < n; ++i) {
    f32 d = t->data->data[i] - mean;
    var += d * d;
  }
  return var / (f32)n;
}

/* Per-(batch, feature) variance across the sequence axis, averaged. */
static f32 position_variance(Tensor t /* [B,S,D] */) {
  u32 B = t->shape[0], S = t->shape[1], D = t->shape[2];
  f32 total = 0;
  for (u32 b = 0; b < B; ++b) {
    for (u32 d = 0; d < D; ++d) {
      f32 mean = 0;
      for (u32 s = 0; s < S; ++s)
        mean += t->data->data[(b * S + s) * D + d];
      mean /= (f32)S;
      f32 var = 0;
      for (u32 s = 0; s < S; ++s) {
        f32 v = t->data->data[(b * S + s) * D + d] - mean;
        var += v * v;
      }
      total += var / (f32)S;
    }
  }
  return total / (f32)(B * D);
}

static void copy_layer_weights(TransformerEncoderLayer dst,
                               TransformerEncoderLayer src) {
  tensor_copy_data(dst->w_qkv, src->w_qkv);
  tensor_copy_data(dst->ffn_0.weight, src->ffn_0.weight);
  tensor_copy_data(dst->ffn_0.bias, src->ffn_0.bias);
  tensor_copy_data(dst->ffn_1.weight, src->ffn_1.weight);
  tensor_copy_data(dst->ffn_1.bias, src->ffn_1.bias);
  tensor_copy_data(dst->layernorm_0.weight, src->layernorm_0.weight);
  tensor_copy_data(dst->layernorm_0.bias, src->layernorm_0.bias);
  tensor_copy_data(dst->layernorm_1.weight, src->layernorm_1.weight);
  tensor_copy_data(dst->layernorm_1.bias, src->layernorm_1.bias);
}

/* ================================================================== */
/* TEST 1 — forward-pass variance with three residual patterns        */
/* ================================================================== */
static void test_forward_variance(void) {
  printf("\n=== TEST 1: forward variance (identical weights + input) ===\n");

  tarena arena = tarena_new(MB(4));
  set_active_arena(arena);

  TransformerEncoderLayer cur = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  TransformerEncoderLayer fix =
      transformer_enc_layer_new(NHEAD, HEAD_DIM, std_pre_ln, std_pre_ln);
  TransformerEncoderLayer pst =
      transformer_enc_layer_new(NHEAD, HEAD_DIM, std_post_ln, std_post_ln);

  copy_layer_weights(fix, cur);
  copy_layer_weights(pst, cur);

  Tensor in = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_SIZE, D_MODEL});
  tensor_random_bound(in, -1.0f, 1.0f);

  Tensor o_cur = transformer_enc_layer_forward(cur, in);
  Tensor o_fix = transformer_enc_layer_forward(fix, in);
  Tensor o_pst = transformer_enc_layer_forward(pst, in);

  printf("pattern               out var      pos var\n");
  printf("current  x + LN(.)   %9.6f   %9.6f\n",
         tensor_variance(o_cur), position_variance(o_cur));
  printf("pre-LN   LN(x) + .   %9.6f   %9.6f\n",
         tensor_variance(o_fix), position_variance(o_fix));
  printf("post-LN  LN(x + .)   %9.6f   %9.6f\n",
         tensor_variance(o_pst), position_variance(o_pst));

  printf("\ninterpretation:\n");
  printf("  pos var ~0  -> output does not distinguish positions\n");
  printf("  out var ~0  -> signal is vanishing\n");

  tarena_destroy(arena);
}

/* ================================================================== */
/* TEST 2 — causal mask correctness                                   */
/* ================================================================== */
static void test_causal_mask(void) {
  printf("\n=== TEST 2: causal mask (perturb last token) ===\n");

  tarena arena = tarena_new(MB(4));
  set_active_arena(arena);

  TransformerEncoderLayer L =
      transformer_enc_layer_new(NHEAD, HEAD_DIM, std_pre_ln, std_pre_ln);

  Tensor x = tensor_new(3, (u32[]){1, SEQ_SIZE, D_MODEL});
  Tensor y = tensor_new(3, (u32[]){1, SEQ_SIZE, D_MODEL});
  tensor_random_bound(x, -1.0f, 1.0f);
  tensor_copy_data(y, x);

  for (u32 d = 0; d < D_MODEL; ++d)
    y->data->data[(SEQ_SIZE - 1) * D_MODEL + d] += 5.0f;

  Tensor ox = transformer_enc_layer_forward(L, x);
  Tensor oy = transformer_enc_layer_forward(L, y);

  f32 d0 = 0, dL = 0;
  for (u32 d = 0; d < D_MODEL; ++d) {
    f32 a0 = ox->data->data[d] - oy->data->data[d];
    f32 aL = ox->data->data[(SEQ_SIZE - 1) * D_MODEL + d]
           - oy->data->data[(SEQ_SIZE - 1) * D_MODEL + d];
    d0 += a0 * a0;
    dL += aL * aL;
  }
  printf("Δ² @ pos 0   (want ~0) : %g\n", d0);
  printf("Δ² @ pos S-1 (want > 0): %g\n", dL);

  if (d0 > 1e-3f)
    printf("[FAIL] future info leaks to position 0\n");
  else if (dL < 1e-3f)
    printf("[FAIL] output ignores its own input\n");
  else
    printf("[OK] causal masking behaves correctly\n");

  tarena_destroy(arena);
}

/* ================================================================== */
/* TEST 3 — gradient magnitudes per parameter                         */
/* ================================================================== */
static void test_gradient_flow(void) {
  printf("\n=== TEST 3: gradient magnitudes (one step, random target) ===\n");

  tarena arena = tarena_new(MB(4));
  set_active_arena(arena);

  TransformerEncoderLayer L =
      transformer_enc_layer_new(NHEAD, HEAD_DIM, std_pre_ln, std_pre_ln);
  transformer_enc_layer_track(L);

  Tensor x = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_SIZE, D_MODEL});
  Tensor y = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_SIZE, D_MODEL});
  tensor_random_bound(x, -1.0f, 1.0f);
  tensor_random_bound(y, -1.0f, 1.0f);

  Tensor out  = transformer_enc_layer_forward(L, x);
  Tensor loss = var_mse_loss(out, y);
  var_backward(loss);

  printf("w_qkv           |g|² = %.6e\n",
         tensor_total_norm_sqr(var_grad(L->w_qkv)));
  printf("ffn_0.weight    |g|² = %.6e\n",
         tensor_total_norm_sqr(var_grad(L->ffn_0.weight)));
  printf("ffn_0.bias      |g|² = %.6e\n",
         tensor_total_norm_sqr(var_grad(L->ffn_0.bias)));
  printf("ffn_1.weight    |g|² = %.6e\n",
         tensor_total_norm_sqr(var_grad(L->ffn_1.weight)));
  printf("layernorm_0.w   |g|² = %.6e\n",
         tensor_total_norm_sqr(var_grad(L->layernorm_0.weight)));

  printf("\ninterpretation:\n");
  printf("  w_qkv should be within ~1-2 orders of magnitude of ffn_0.\n");
  printf("  If w_qkv is 1e-8 while ffn_0 is 1e-2, attention is starved.\n");

  tarena_destroy(arena);
}

/* ================================================================== */
/* TEST 4 — head-to-head training, current vs std pre-LN              */
/* ================================================================== */
static void test_train_ab(void) {
  printf("\n=== TEST 4: train current vs std pre-LN, identical init ===\n");

  tarena param_arena = tarena_new(MB(4));
  tarena step_arena  = tarena_new(MB(8));

  set_active_arena(param_arena);

  TransformerEncoderLayer cur = transformer_enc_layer_new(
      NHEAD, HEAD_DIM, pre_ln_residual, pre_ln_residual);
  TransformerEncoderLayer fix =
      transformer_enc_layer_new(NHEAD, HEAD_DIM, std_pre_ln, std_pre_ln);
  copy_layer_weights(fix, cur);

  transformer_enc_layer_track(cur);
  transformer_enc_layer_track(fix);

  Tensor p_cur[N_XFORMER_KERNELS];
  Tensor p_fix[N_XFORMER_KERNELS];
  u32 n_cur =
      transformer_enc_layer_kernels(cur, p_cur, N_XFORMER_KERNELS);
  u32 n_fix =
      transformer_enc_layer_kernels(fix, p_fix, N_XFORMER_KERNELS);

  SgdOptimizer opt_cur =
      sgd_optimizer_new(p_cur, n_cur, LR, GRAD_THR, VAL_CLAMP);
  SgdOptimizer opt_fix =
      sgd_optimizer_new(p_fix, n_fix, LR, GRAD_THR, VAL_CLAMP);

  /* inputs live in param_arena so they survive step_arena resets */
  Tensor x = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_SIZE, D_MODEL});
  Tensor y = tensor_new(3, (u32[]){BATCH_SIZE, SEQ_SIZE, D_MODEL});
  tensor_random_bound(x, -1.0f, 1.0f);
  tensor_random_bound(y, -1.0f, 1.0f);

  printf("step | current   | std pre-LN\n");
  printf("-----+-----------+------------\n");

  for (u32 step = 0; step <= 200; ++step) {
    set_active_arena(step_arena);

    Tensor oc = transformer_enc_layer_forward(cur, x);
    Tensor of = transformer_enc_layer_forward(fix, x);
    Tensor lc = var_mse_loss(oc, y);
    Tensor lf = var_mse_loss(of, y);

    if (step % 20 == 0)
      printf("%4u | %9.6f | %9.6f\n", step, lc->data->data[0],
             lf->data->data[0]);

    var_backward(lc);
    var_backward(lf);

    sgd_optimize(opt_cur, nullptr);
    sgd_optimize(opt_fix, nullptr);

    var_zero_grad(lc);
    var_zero_grad(lf);

    tarena_reset(step_arena);
  }

  sgd_optimizer_destroy(opt_cur);
  sgd_optimizer_destroy(opt_fix);
  tarena_destroy(param_arena);
  tarena_destroy(step_arena);
}

/* ================================================================== */

int main(void) {
  printf("Transformer plateau diagnostics\n");
  printf("===============================\n");

  test_forward_variance();
  test_causal_mask();
  test_gradient_flow();
  test_train_ab();

  printf("\nDiagnostics complete.\n");
  return 0;
}
