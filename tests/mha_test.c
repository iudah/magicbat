#include "type_alias.h"
#include <stdio.h>
#include <stdlib.h>
void test_fused_mha(u32 seq_len);
void test_discrete_mha(u32 seq_len);
int main() {
  printf("\nFused Test: 8\n");
  test_fused_mha(8);
  printf("\nFused Test: 32\n");
  test_fused_mha(32);
  return 0;
}
