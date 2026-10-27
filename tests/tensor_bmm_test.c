#include "tensor.h"
#include "var.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-4f

#define ASSERT_NEAR(a, b, msg)                                                 \
  do {                                                                         \
    if (fabsf((a) - (b)) > EPS) {                                              \
      fprintf(stderr, "FAIL: %s | %f != %f at %s:%d\n", (msg), (a), (b),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static void print_tensor_summary(const char *name, Tensor t) {
  if (!t) {
    printf("%s: nullptr\n", name);
    return;
  }
  printf("%s: shape=[", name);
  for (u32 i = 0; i < t->ndims; ++i)
    printf("%u%s", t->shape[i], i < t->ndims - 1 ? "," : "");
  printf("] nelem=%u cont=%d\n", tensor_num_elements(t), t->is_contiguous);
}

static Tensor make_tensor(u32 ndims, const u32 *shape, const f32 *data) {
  Tensor t = tensor_new(ndims, shape);
  if (t && data) {
    memcpy(t->data->data, data, tensor_num_elements(t) * sizeof(f32));
  }
  return t;
}

static void test_basic_bmm(void) {
  printf("  Basic 3D BMM...\n");
  u32 shape_a[] = {2, 2, 3};
  u32 shape_b[] = {2, 3, 4};
  f32 a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  f32 b_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

  Tensor a = make_tensor(3, shape_a, a_data);
  Tensor b = make_tensor(3, shape_b, b_data);
  Tensor res = tensor_bmm(a, b);

  assert(res && res->ndims == 3 && res->shape[0] == 2 && res->shape[1] == 2 &&
         res->shape[2] == 4);

  // First batch first row first elem: 1*1 + 2*5 + 3*9 = 38
  ASSERT_NEAR(res->data->data[0], 38.0f, "basic[0,0,0]");
  ASSERT_NEAR(res->data->data[1], 44.0f, "basic[0,0,1]");

  tensor_destroy(a);
  tensor_destroy(b);
  tensor_destroy(res);
}

static void test_broadcasting(void) {
  printf("  Broadcasting (batch=1)...\n");
  u32 shape_a[] = {1, 2, 3}; // broadcast batch
  u32 shape_b[] = {2, 3, 4};
  f32 a_data[] = {1, 2, 3, 4, 5, 6};
  f32 b_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

  Tensor a = make_tensor(3, shape_a, a_data);
  Tensor b = make_tensor(3, shape_b, b_data);
  Tensor res = tensor_bmm(a, b);

  assert(res && res->shape[0] == 2);
  ASSERT_NEAR(res->data->data[0], 38.0f, "broadcast[0]");

  tensor_destroy(a);
  tensor_destroy(b);
  tensor_destroy(res);
}

static void test_2d_as_batched(void) {
  printf("  2D treated as batched...\n");
  u32 shape_a[] = {2, 3};
  u32 shape_b[] = {3, 4};
  f32 a_data[] = {1, 2, 3, 4, 5, 6};
  f32 b_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  Tensor a = make_tensor(2, shape_a, a_data);
  Tensor b = make_tensor(2, shape_b, b_data);
  Tensor res = tensor_bmm(a, b);

  assert(res && res->ndims == 3 && res->shape[0] == 1);
  tensor_destroy(a);
  tensor_destroy(b);
  tensor_destroy(res);
}

static void test_transpose_variants(void) {
  printf("  Transpose B / Transpose A...\n");
  u32 s2[] = {2, 2};
  f32 a_data[] = {1, 2, 3, 4};
  f32 b_data[] = {5, 6, 7, 8};

  Tensor a = make_tensor(2, s2, a_data);
  Tensor b = make_tensor(2, s2, b_data);

  Tensor res_tb = tensor_bmatmul_transpose_b(a, b); // A @ B^T
  Tensor res_ta = tensor_bmatmul_transpose_a(a, b); // A^T @ B

  assert(res_tb && res_ta);
  // Quick sanity check (exact values depend on data)
  ASSERT_NEAR(res_ta->data->data[0], 1. * 5. + 3. * 7., "transpose_a[0]");
  ASSERT_NEAR(res_tb->data->data[0], 1. * 5. + 2. * 6., "transpose_b[0]");

  tensor_destroy(a);
  tensor_destroy(b);
  tensor_destroy(res_tb);
  tensor_destroy(res_ta);
}

static void test_autograd_bmm(void) {
  printf("  Autograd (var_bmm + backward)...\n");

  Tensor a_raw = tensor_new(3, (u32[]){1, 2, 3});
  Tensor b_raw = tensor_new(3, (u32[]){1, 3, 2});

  // Fill with simple values
  for (u32 i = 0; i < tensor_num_elements(a_raw); ++i)
    a_raw->data->data[i] = (f32)(i + 1);
  for (u32 i = 0; i < tensor_num_elements(b_raw); ++i)
    b_raw->data->data[i] = (f32)(i + 1);

  a_raw->requires_grad = true;
  b_raw->requires_grad = true;

  Tensor out = var_bmm(a_raw, b_raw);
  assert(out);

  Tensor loss_grad = tensor_scalar(1.0f);
  var_backward_with_grad(out, loss_grad);

  assert(a_raw->grad != nullptr);
  assert(b_raw->grad != nullptr);

  tensor_destroy(a_raw);
  tensor_destroy(b_raw);
  tensor_destroy(out);
  tensor_destroy(loss_grad);
}

int main(void) {
  printf("=== BMM Unit Test Suite ===\n\n");

  test_basic_bmm();
  test_broadcasting();
  test_2d_as_batched();
  test_transpose_variants();
  test_autograd_bmm();

  printf("\nAll BMM tests passed!\n");
  return 0;
}
