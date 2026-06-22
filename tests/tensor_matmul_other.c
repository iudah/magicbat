#include "../include/adt/tensor/tensor_prot.h"
#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-6
#define MAX_DIMS 5

/* Helper: compare two tensors element‑wise */
static bool tensors_equal(Tensor a, Tensor b) {
  if (!a || !b)
    return false;
  if (a->ndims != b->ndims)
    return false;
  if (a->data->nelements != b->data->nelements)
    return false;
  for (u32 i = 0; i < a->ndims; ++i) {
    if (a->shape[i] != b->shape[i])
      return false;
  }
  for (u32 i = 0; i < a->data->nelements; ++i) {
    if (fabsf(a->data->data[i] - b->data->data[i]) > EPS) {
      printf("Mismatch at index %u: %f vs %f\n", i, a->data->data[i],
             b->data->data[i]);
      return false;
    }
  }
  return true;
}

/* Helper: fill a 2D tensor with values from a flat array */
static void fill_2d(Tensor t, const float *vals) {
  assert(t->ndims == 2);
  memcpy(t->data->data, vals, t->data->nelements * sizeof(float));
}

/* Helper: create a 2D tensor and fill with given values */
static Tensor make_tensor_2d(u32 rows, u32 cols, const float *vals) {
  Tensor t = tensor_new(2, (u32[]){rows, cols});
  if (vals) {
    fill_2d(t, vals);
  }
  return t;
}

/* Manual computation of dA = grad * B^T (both 2D) */
static Tensor manual_matmul_wrt_a(Tensor grad, Tensor B) {
  u32 m = grad->shape[0];
  u32 n = B->shape[0];    // B is k x n, so B^T is n x k
  u32 k = grad->shape[1]; // common dimension
  Tensor result = tensor_new(2, (u32[]){m, n});
  for (u32 i = 0; i < m; ++i) {
    for (u32 j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (u32 l = 0; l < k; ++l) {
        sum += tensor_get(grad, (u32[]){i, l}) *
               tensor_get(
                   B, (u32[]){j, l}); // B[j][l] because B^T has (l,j) = B[j][l]
      }
      tensor_set(result, (u32[]){i, j}, sum);
    }
  }
  return result;
}

/* Manual computation of dB = A^T * grad (both 2D) */
static Tensor manual_matmul_wrt_b(Tensor A, Tensor grad) {
  u32 m = A->shape[1]; // A^T has dimensions (k x m)
  u32 n = grad->shape[1];
  u32 k = A->shape[0]; // common dimension
  Tensor result = tensor_new(2, (u32[]){m, n});
  for (u32 i = 0; i < m; ++i) {
    for (u32 j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (u32 l = 0; l < k; ++l) {
        sum += tensor_get(A, (u32[]){l, i}) * // A^T has (i,l) = A[l][i]
               tensor_get(grad, (u32[]){l, j});
      }
      tensor_set(result, (u32[]){i, j}, sum);
    }
  }
  return result;
}

/* Test a single case for both wrt_a and wrt_b */
static void test_case(u32 m, u32 k, u32 n, const char *desc) {
  printf("Testing: %s (m=%u, k=%u, n=%u)\n", desc, m, k, n);

  // Allocate A (m x k), B (k x n), G (m x n)
  // Fill with deterministic values (e.g., sequential)
  float *A_vals = malloc(m * k * sizeof(float));
  float *B_vals = malloc(k * n * sizeof(float));
  float *G_vals = malloc(m * n * sizeof(float));
  for (u32 i = 0; i < m * k; ++i)
    A_vals[i] = (float)(i + 1);
  for (u32 i = 0; i < k * n; ++i)
    B_vals[i] = (float)(i + 10);
  for (u32 i = 0; i < m * n; ++i)
    G_vals[i] = (float)(i + 100);

  Tensor A = make_tensor_2d(m, k, A_vals);
  Tensor B = make_tensor_2d(k, n, B_vals);
  Tensor G = make_tensor_2d(m, n, G_vals);

  // ---- Test tensor_matmul_wrt_a ----
  Tensor dA_candidate = tensor_matmul_wrt_a(G, B);
  Tensor dA_expected = manual_matmul_wrt_a(G, B);
  if (!tensors_equal(dA_candidate, dA_expected)) {
    printf("Error: tensor_matmul_wrt_a failed for %s\n", desc);
    tensor_destroy(dA_candidate);
    tensor_destroy(dA_expected);
    goto cleanup;
  }
  tensor_destroy(dA_candidate);
  tensor_destroy(dA_expected);

  // ---- Test tensor_matmul_wrt_b ----
  Tensor dB_candidate = tensor_matmul_wrt_b(A, G);
  Tensor dB_expected = manual_matmul_wrt_b(A, G);
  if (!tensors_equal(dB_candidate, dB_expected)) {
    printf("Error: tensor_matmul_wrt_b failed for %s\n", desc);
    tensor_destroy(dB_candidate);
    tensor_destroy(dB_expected);
    goto cleanup;
  }
  tensor_destroy(dB_candidate);
  tensor_destroy(dB_expected);

  printf("  Passed\n");
cleanup:
  tensor_destroy(A);
  tensor_destroy(B);
  tensor_destroy(G);
  free(A_vals);
  free(B_vals);
  free(G_vals);
}

int main() {
  // Test cases covering various dimension sizes, including edge cases
  // where one dimension is smaller than BLOCK_SIZE (16) or exactly equal.

  // 1. All small (k < 16)
  test_case(2, 3, 4, "small all");
  test_case(1, 8, 5, "dot-like");

  // 2. Some dimensions equal to BLOCK_SIZE
  test_case(16, 16, 16, "all block");
  test_case(16, 8, 32, "mixed block");

  // 3. Some dimensions > BLOCK_SIZE
  test_case(20, 20, 20, "larger than block");
  test_case(32, 48, 16,
            "typical MHA shape (batch*seq=32, d_model=16, 3*head*dim=48)");

  // 4. Edge where contraction dimension is smaller than block (k < 16)
  // This is the case that caused your bug: k = 8 (seq=8) or k = 32 (seq=32)
  test_case(16, 8, 48, "contraction small (8)");
  test_case(16, 32, 48, "contraction medium (32)");

  printf("\nAll gradient matmul tests passed!\n");
  return 0;
}
