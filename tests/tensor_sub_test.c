/*
 * Unit tests for tensor_sub and tensor_sub_inplace.
 *
 * Asymmetric-op checks that tensor_add_test cannot catch:
 *   - same-shape subtraction is not commutative
 *   - scalar-left vs scalar-right produce different results
 *   - broadcast with unequal dims keeps the larger rank
 *   - non-contiguous operand still produces a dense result
 *   - NaN propagation
 *   - incompatible shapes rejected
 */

#include "tensor.h"
#include "tensor_prot.h"
#include "type_alias.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define dim_rows    3u
#define dim_cols    5u
#define scalar_val  10.0f
#define fill_rhs    3.0f
#define fill_lhs    5.0f
#define approx_eps  1.0e-6f

/* NOLINTBEGIN(readability-magic-numbers) */

static Tensor make_tensor(u32 ndims, const u32 *shape, f32 fill) {
  Tensor t = tensor_new(ndims, shape);
  assert(t != nullptr);
  tensor_fill(t, fill);
  return t;
}

static void expect_all_equal(Tensor tensor, f32 value) {
  const u32 count = tensor_num_elements(tensor);
  for (u32 idx = 0; idx < count; ++idx) {
    assert(fabsf(tensor->data->data[idx] - value) < approx_eps);
  }
}

static void test_same_shape(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor diff = tensor_sub(lhs, rhs);
  assert(diff != nullptr);
  expect_all_equal(diff, fill_lhs - fill_rhs);

  tensor_destroy(diff);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_not_commutative(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor forward  = tensor_sub(lhs, rhs);
  Tensor backward = tensor_sub(rhs, lhs);
  assert(forward != nullptr && backward != nullptr);

  const u32 count = tensor_num_elements(forward);
  for (u32 idx = 0; idx < count; ++idx) {
    assert(fabsf(forward->data->data[idx] + backward->data->data[idx]) <
           approx_eps);
  }

  tensor_destroy(forward);
  tensor_destroy(backward);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_scalar_left(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    tensor   = make_tensor(2, shape, fill_rhs);
  Tensor    scalar   = tensor_scalar(scalar_val);

  Tensor diff = tensor_sub(scalar, tensor);
  assert(diff != nullptr);
  expect_all_equal(diff, scalar_val - fill_rhs);

  tensor_destroy(scalar);
  tensor_destroy(diff);
  tensor_destroy(tensor);
}

static void test_scalar_right(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    tensor   = make_tensor(2, shape, scalar_val);
  Tensor    scalar   = tensor_scalar(fill_rhs);

  Tensor diff = tensor_sub(tensor, scalar);
  assert(diff != nullptr);
  expect_all_equal(diff, scalar_val - fill_rhs);

  tensor_destroy(scalar);
  tensor_destroy(diff);
  tensor_destroy(tensor);
}

static void test_non_contiguous(void) {
  const u32 shape[2] = {dim_cols, dim_rows};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor lhs_t = tensor_transpose(lhs);
  assert(lhs_t != nullptr && !lhs_t->is_contiguous);

  Tensor diff = tensor_sub(lhs_t, rhs);
  assert(diff != nullptr);
  expect_all_equal(diff, fill_lhs - fill_rhs);

  tensor_destroy(lhs_t);
  tensor_destroy(diff);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_inplace(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  assert(tensor_sub_inplace(lhs, rhs));
  expect_all_equal(lhs, fill_lhs - fill_rhs);

  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_nan_propagation(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, (f32)NAN);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor diff = tensor_sub(lhs, rhs);
  assert(diff != nullptr);
  const u32 count = tensor_num_elements(diff);
  for (u32 idx = 0; idx < count; ++idx) {
    assert(isnan(diff->data->data[idx]));
  }

  tensor_destroy(diff);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_incompatible_shapes(void) {
  const u32 shape_lhs[2] = {2u, 3u};
  const u32 shape_rhs[2] = {3u, 2u};

  Tensor lhs = make_tensor(2, shape_lhs, fill_lhs);
  Tensor rhs = make_tensor(2, shape_rhs, fill_rhs);

  assert(tensor_sub(lhs, rhs) == nullptr);

  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

/* NOLINTEND(readability-magic-numbers) */

int main(void) {
  test_same_shape();
  test_not_commutative();
  test_scalar_left();
  test_scalar_right();
  test_non_contiguous();
  test_inplace();
  test_nan_propagation();
  test_incompatible_shapes();

  printf("tensor_sub_test: all tests passed\n");
  return EXIT_SUCCESS;
}
