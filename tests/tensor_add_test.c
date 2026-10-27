/*
 * Unit tests for tensor_add, tensor_add_inplace, tensor_scaled_add and
 * tensor_add_scalar_inplace.
 *
 * Exercises:
 *   - same-shape addition, contiguous and non-contiguous operands
 *   - scalar broadcast, both operand orders
 *   - general broadcasting, rank-equal and rank-raising
 *   - in-place addition with aliasing rules respected
 *   - scaled addition   a + alpha * b
 *   - scalar in-place addition
 *   - NaN propagation (IEEE-754: NaN + x == NaN)
 *   - incompatible shapes are rejected
 */

#include "tensor.h"
#include "tensor_prot.h"
#include "type_alias.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define dim_rows       2u
#define dim_cols       3u
#define bcast_rows     3u
#define bcast_cols     4u

#define fill_lhs       2.5f
#define fill_rhs       6.5f
#define probe_value    45.23f
#define scalar_value   10.0f
#define scaled_alpha  (-0.5f)

#define approx_eps     1.0e-6f

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

static void expect_all_nan(Tensor tensor) {
  const u32 count = tensor_num_elements(tensor);
  for (u32 idx = 0; idx < count; ++idx) {
    assert(isnan(tensor->data->data[idx]));
  }
}

static void test_same_shape_contiguous(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor sum = tensor_add(lhs, rhs);
  assert(sum != nullptr);
  assert(tensor_ndims(sum) == 2);
  expect_all_equal(sum, fill_lhs + fill_rhs);

  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_same_shape_non_contiguous(void) {
  const u32 shape[2] = {bcast_rows, dim_rows};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor lhs_t = tensor_transpose(lhs);
  assert(lhs_t != nullptr && !lhs_t->is_contiguous);

  Tensor sum = tensor_add(lhs_t, rhs);
  assert(sum != nullptr);
  expect_all_equal(sum, fill_lhs + fill_rhs);

  tensor_destroy(lhs_t);
  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_scalar_left(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    tensor   = make_tensor(2, shape, fill_lhs);
  Tensor    scalar   = tensor_scalar(scalar_value);

  Tensor sum = tensor_add(scalar, tensor);
  assert(sum != nullptr);
  expect_all_equal(sum, scalar_value + fill_lhs);

  tensor_destroy(scalar);
  tensor_destroy(sum);
  tensor_destroy(tensor);
}

static void test_scalar_right(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    tensor   = make_tensor(2, shape, fill_lhs);
  Tensor    scalar   = tensor_scalar(scalar_value);

  Tensor sum = tensor_add(tensor, scalar);
  assert(sum != nullptr);
  expect_all_equal(sum, fill_lhs + scalar_value);

  tensor_destroy(scalar);
  tensor_destroy(sum);
  tensor_destroy(tensor);
}

static void test_general_broadcast(void) {
  const u32 shape_lhs[2] = {bcast_rows, 1u};
  const u32 shape_rhs[2] = {bcast_rows, bcast_cols};

  Tensor lhs = make_tensor(2, shape_lhs, 10.0f);
  Tensor rhs = make_tensor(2, shape_rhs, 1.0f);

  tensor_set(lhs, (u32[]){1u, 0u}, 100.0f);

  Tensor sum = tensor_add(lhs, rhs);
  assert(sum != nullptr);
  assert(tensor_shape(sum)[0] == bcast_rows);
  assert(tensor_shape(sum)[1] == bcast_cols);

  assert(fabsf(tensor_get(sum, (u32[]){0u, 0u}) - 11.0f) < approx_eps);
  assert(fabsf(tensor_get(sum, (u32[]){1u, 2u}) - 101.0f) < approx_eps);
  assert(fabsf(tensor_get(sum, (u32[]){2u, 3u}) - 11.0f) < approx_eps);

  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_inplace_contiguous(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  assert(tensor_add_inplace(lhs, rhs));
  expect_all_equal(lhs, fill_lhs + fill_rhs);

  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_scaled_add(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor sum = tensor_scaled_add(lhs, scaled_alpha, rhs);
  assert(sum != nullptr);
  expect_all_equal(sum, fill_lhs + scaled_alpha * fill_rhs);

  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_add_scalar_inplace(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    tensor   = make_tensor(2, shape, fill_lhs);

  assert(tensor_add_scalar_inplace(tensor, scalar_value));
  expect_all_equal(tensor, fill_lhs + scalar_value);

  tensor_destroy(tensor);
}

static void test_nan_propagation(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, (f32)NAN);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  Tensor sum = tensor_add(lhs, rhs);
  assert(sum != nullptr);
  expect_all_nan(sum);

  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_probe_value_preserved(void) {
  const u32 shape[2] = {dim_rows, dim_cols};
  Tensor    lhs      = make_tensor(2, shape, fill_lhs);
  Tensor    rhs      = make_tensor(2, shape, fill_rhs);

  tensor_set(lhs, (u32[]){1u, 2u}, probe_value);

  Tensor sum = tensor_add(lhs, rhs);
  assert(sum != nullptr);
  assert(fabsf(tensor_get(sum, (u32[]){1u, 2u}) -
               (probe_value + fill_rhs)) < approx_eps);

  tensor_destroy(sum);
  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

static void test_incompatible_shapes(void) {
  const u32 shape_lhs[2] = {2u, 3u};
  const u32 shape_rhs[2] = {3u, 2u};

  Tensor lhs = make_tensor(2, shape_lhs, fill_lhs);
  Tensor rhs = make_tensor(2, shape_rhs, fill_rhs);

  assert(tensor_add(lhs, rhs) == nullptr);

  tensor_destroy(lhs);
  tensor_destroy(rhs);
}

/* NOLINTEND(readability-magic-numbers) */

int main(void) {
  test_same_shape_contiguous();
  test_same_shape_non_contiguous();
  test_scalar_left();
  test_scalar_right();
  test_general_broadcast();
  test_inplace_contiguous();
  test_scaled_add();
  test_add_scalar_inplace();
  test_nan_propagation();
  test_probe_value_preserved();
  test_incompatible_shapes();

  printf("tensor_add_test: all tests passed\n");
  return EXIT_SUCCESS;
}
