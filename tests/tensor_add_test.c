#include "../include/tensor/adt/tensor_prot.h"
#include "../include/tensor/tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main() {
  Tensor t = tensor_new(2, (u32[]){3, 5});
  Tensor s = tensor_new(2, (u32[]){3, 5});

  assert(t != NULL);
  assert(s != NULL);

  tensor_fill(t, 2.5f);
  tensor_fill(s, 6.5f);

  tensor_set(t, (u32[]){1, 3}, 45.23f);
  tensor_set(s, (u32[]){2, 2}, -5.55f);

  Tensor sum = tensor_add(t, s);
  assert(sum != NULL);

  for (u32 i = 0; i < sum->nelements; ++i) {
    assert(sum->data[i] == (t->data[i] + s->data[i]));
  }

  tensor_destroy(sum);
  tensor_destroy(s);
  tensor_destroy(t);

  printf("All add tests passed\n");

  return 0;
}