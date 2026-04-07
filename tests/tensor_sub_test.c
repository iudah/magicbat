#include "../include/adt/tensor/tensor_prot.h"
#include "../include/tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main() {
  /* ---------------- SAME SHAPE ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){3, 5});
    Tensor s = tensor_new(2, (u32[]){3, 5});

    assert(t && s);

    tensor_fill(t, 5.0f);
    tensor_fill(s, 2.0f);

    Tensor res = tensor_sub(t, s);
    assert(res);

    for (u32 i = 0; i < res->data->nelements; ++i)
      assert(res->data->data[i] == 3.0f);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- SCALAR ---------------- */
  {
    Tensor t = tensor_new(1, (u32[]){1});
    Tensor s = tensor_new(2, (u32[]){2, 2});

    t->data->data[0] = 10.0f;
    tensor_fill(s, 3.0f);

    Tensor res = tensor_sub(t, s);
    assert(res);

    for (u32 i = 0; i < res->data->nelements; ++i)
      assert(res->data->data[i] == 7.0f);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- INVALID ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    Tensor s = tensor_new(2, (u32[]){3, 2});

    Tensor res = tensor_sub(t, s);
    assert(res == NULL);

    tensor_destroy(t);
    tensor_destroy(s);
  }

  printf("All sub tests passed\n");
  return 0;
}