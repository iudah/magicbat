#include "tensor_prot.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define EPS 1e-6

int main() {
  /* ---------------- SAME SHAPE ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 2});
    Tensor s = tensor_new(2, (u32[]){2, 2});

    tensor_fill(t, 10.0f);
    tensor_fill(s, 2.0f);

    Tensor res = tensor_div(t, s);
    assert(res);

    for (u32 i = 0; i < res->data->nelements; ++i)
      assert(fabsf(res->data->data[i] - 5.0f) < EPS);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- SCALAR ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 2});
    Tensor s = tensor_new(1, (u32[]){1});

    tensor_fill(t, 9.0f);
    s->data->data[0] = 3.0f;

    Tensor res = tensor_div(t, s);
    assert(res);

    for (u32 i = 0; i < res->data->nelements; ++i)
      assert(fabsf(res->data->data[i] - 3.0f) < EPS);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- INVALID ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    Tensor s = tensor_new(2, (u32[]){3, 2});

    Tensor res = tensor_div(t, s);
    assert(res == nullptr);

    tensor_destroy(t);
    tensor_destroy(s);
  }

  printf("All div tests passed\n");
  return 0;
}
