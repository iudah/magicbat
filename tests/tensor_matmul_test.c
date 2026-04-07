#include "../include/adt/tensor/tensor_prot.h"
#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define EPS 1e-6

int main() {
  /* ---------------- GENERAL CASE ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    Tensor s = tensor_new(2, (u32[]){3, 2});

    assert(t && s);

    // t = [1 2 3
    //      4 5 6]
    float t_vals[] = {1, 2, 3, 4, 5, 6};
    for (u32 i = 0; i < 6; ++i)
      t->data->data[i] = t_vals[i];

    // s = [7  8
    //      9 10
    //     11 12]
    float s_vals[] = {7, 8, 9, 10, 11, 12};
    for (u32 i = 0; i < 6; ++i)
      s->data->data[i] = s_vals[i];

    Tensor res = tensor_matmul(t, s);
    assert(res);

    // Expected:
    // [58  64
    // 139 154]
    float expected[] = {58, 64, 139, 154};

    for (u32 i = 0; i < 4; ++i) {
      assert(fabsf(res->data->data[i] - expected[i]) < EPS);
    }

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- IDENTITY SCALING ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){3, 3});
    Tensor s = tensor_new(2, (u32[]){3, 2});

    assert(t && s);

    tensor_fill(t, 0.0f);
    for (u32 i = 0; i < 3; ++i) {
      tensor_set(t, (u32[]){i, i}, 3.0f);
    }

    tensor_fill(s, 2.0f);

    Tensor res = tensor_matmul(t, s);
    assert(res);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- DOT PRODUCT (1xN * Nx1) ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){1, 3});
    Tensor s = tensor_new(2, (u32[]){3, 1});

    assert(t && s);

    // t = [1 2 3]
    float t_vals[] = {1, 2, 3};
    for (u32 i = 0; i < 3; ++i)
      t->data->data[i] = t_vals[i];

    // s = [4
    //      5
    //      6]
    float s_vals[] = {4, 5, 6};
    for (u32 i = 0; i < 3; ++i)
      s->data->data[i] = s_vals[i];

    Tensor res = tensor_matmul(t, s);
    assert(res);

    // Expected: 1*4 + 2*5 + 3*6 = 32
    assert(fabsf(res->data->data[0] - 32.0f) < EPS);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- SHAPE MISMATCH ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    Tensor s = tensor_new(2, (u32[]){4, 2});

    assert(t && s);

    Tensor res = tensor_matmul(t, s);
    assert(res == NULL);

    tensor_destroy(t);
    tensor_destroy(s);
  }

  printf("All matmul tests passed\n");
  return 0;
}