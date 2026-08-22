#include "tensor_memory.h"
int main() {
  tfree(tmalloc(100));
  return 0;
}
