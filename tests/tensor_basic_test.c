#include "../include/tensor/adt/tensor_prot.h"
#include "../include/tensor/tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main() {
  Tensor t = tensor_new(2, (u32[]){3, 5});

  assert(t != NULL && "tensor_new returned NULL");
  assert(t->ndims == 2);
  assert(t->shape[0] == 3);
  assert(t->shape[1] == 5);
  assert(t->data != NULL);
  assert(t->nelements == 15);

  tensor_destroy(t);

  printf("All tests passed");

  return 0;
}