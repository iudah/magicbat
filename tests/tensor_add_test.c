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

    tensor_fill(t, 2.5f);
    tensor_fill(s, 6.5f);

    tensor_set(t, (u32[]){1, 3}, 45.23f);
    tensor_set(s, (u32[]){2, 2}, -5.55f);

    Tensor sum = tensor_add(t, s);
    assert(sum);

    for (u32 i = 0; i < sum->nelements; ++i) {
      assert(sum->data[i] == (t->data[i] + s->data[i]));
    }

    tensor_destroy(sum);
    tensor_destroy(s);
    tensor_destroy(t);
  }

  /* ---------------- SCALAR BROADCAST (t + scalar) ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 2});
    Tensor s = tensor_new(1, (u32[]){1});

    assert(t && s);

    float vals[] = {1, 2, 3, 4};
    for (u32 i = 0; i < 4; ++i)
      t->data[i] = vals[i];

    s->data[0] = 10.0f;

    Tensor res = tensor_add(t, s);
    assert(res);

    float expected[] = {11, 12, 13, 14};

    for (u32 i = 0; i < 4; ++i)
      assert(res->data[i] == expected[i]);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- SCALAR BROADCAST (scalar + t) ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 2});
    Tensor s = tensor_new(1, (u32[]){1});

    assert(t && s);

    float vals[] = {5, 6, 7, 8};
    for (u32 i = 0; i < 4; ++i)
      t->data[i] = vals[i];

    s->data[0] = 2.0f;

    Tensor res = tensor_add(s, t);
    assert(res);

    float expected[] = {7, 8, 9, 10};

    for (u32 i = 0; i < 4; ++i)
      assert(res->data[i] == expected[i]);

    tensor_destroy(res);
    tensor_destroy(t);
    tensor_destroy(s);
  }

  /* ---------------- INVALID SHAPE ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    Tensor s = tensor_new(2, (u32[]){3, 2});

    assert(t && s);

    Tensor res = tensor_add(t, s);
    assert(res == NULL);

    tensor_destroy(t);
    tensor_destroy(s);
  }

  printf("All add tests passed\n");
  return 0;
}