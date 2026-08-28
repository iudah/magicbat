#include "tensor_prot.h"
#include "tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main() {
  Tensor t = tensor_new(2, (u32[]){3, 5});

  assert(t != nullptr && "tensor_new returned nullptr");
  assert(t->ndims == 2);
  assert(t->shape[0] == 3);
  assert(t->shape[1] == 5);
  assert(t->data != nullptr);
  assert(t->data->nelements == 15);

  tensor_destroy(t);

  printf("All basic tests passed\n");

  return 0;
}
