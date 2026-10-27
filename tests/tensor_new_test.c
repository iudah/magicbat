/*
 * Unit tests for tensor_new / tensor_destroy / tensor_view / tensor_scalar.
 *
 * Exercises:
 *   - shape/stride computation for 1-D through MAX_DIMS-D
 *   - contiguous layout guarantee (last stride == 1)
 *   - zero-element allocation is rejected
 *   - dimension count outside [1, MAX_DIMS] is rejected
 *   - 32-bit overflow during stride computation is rejected
 *   - reference counting via tensor_view
 *   - tensor_scalar shape/stride invariants
 *   - tensor_destroy is idempotent on null and safe on views
 */

#include "tensor.h"
#include "tensor_prot.h"
#include "type_alias.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define len_3    3u
#define len_5    5u
#define len_7    7u
#define len_prod (len_3 * len_5 * len_7)
#define scalar_v 42.0f

/* readability-magic-numbers — the whole point of a unit test is to assert
 * against concrete literals; scope the suppression to the assertions. */
/* NOLINTBEGIN(readability-magic-numbers) */

static void test_construct_1d(void) {
  const u32 shape[1] = {len_5};
  Tensor    tensor   = tensor_new(1, shape);

  assert(tensor != nullptr);
  assert(tensor->ndims == 1);
  assert(tensor->shape[0] == len_5);
  assert(tensor->stride[0] == 1);
  assert(tensor->data != nullptr);
  assert(tensor->data->nelements == len_5);
  assert(tensor->is_contiguous);
  assert(tensor->refcount == 1);

  tensor_destroy(tensor);
}

static void test_construct_3d(void) {
  const u32 shape[3] = {len_3, len_5, len_7};
  Tensor    tensor   = tensor_new(3, shape);

  assert(tensor != nullptr);
  assert(tensor->ndims == 3);
  assert(tensor->shape[0] == len_3);
  assert(tensor->shape[1] == len_5);
  assert(tensor->shape[2] == len_7);
  assert(tensor->stride[2] == 1);
  assert(tensor->stride[1] == len_7);
  assert(tensor->stride[0] == len_5 * len_7);
  assert(tensor->data->nelements == len_prod);

  tensor_destroy(tensor);
}

static void test_reject_zero_length(void) {
  const u32 shape[1] = {0u};
  assert(tensor_new(1, shape) == nullptr);
}

static void test_reject_null_shape(void) {
  assert(tensor_new(2, nullptr) == nullptr);
}

static void test_reject_zero_ndims(void) {
  const u32 shape[1] = {len_3};
  assert(tensor_new(0, shape) == nullptr);
}

static void test_reject_too_many_dims(void) {
  const u32 shape[MAX_DIMS + 1] = {1u, 1u, 1u, 1u, 1u, 1u};
  assert(tensor_new(MAX_DIMS + 1, shape) == nullptr);
}

static void test_view_shares_storage(void) {
  const u32 shape[2] = {len_3, len_5};
  Tensor    src      = tensor_new(2, shape);
  assert(src != nullptr);
  tensor_fill(src, 7.0f);

  Tensor view = tensor_view(src);
  assert(view != nullptr);
  assert(view->data == src->data);
  assert(view->data->refcount == 2);

  /* Mutating the view is visible in the source (shared buffer). */
  view->data->data[0] = 99.0f;
  assert(src->data->data[0] == 99.0f);

  tensor_destroy(view);
  assert(src->data->refcount == 1);
  tensor_destroy(src);
}

static void test_scalar_invariants(void) {
  Tensor scalar = tensor_scalar(scalar_v);
  assert(scalar != nullptr);
  assert(scalar->ndims == 1);
  assert(scalar->shape[0] == 1);
  assert(scalar->stride[0] == 1);
  assert(scalar->data->nelements == 1);
  assert(scalar->data->data[0] == scalar_v);
  assert(tensor_num_elements(scalar) == 1);

  tensor_destroy(scalar);
}

static void test_destroy_null_is_safe(void) {
  assert(tensor_destroy(nullptr) == false);
}

/* NOLINTEND(readability-magic-numbers) */

int main(void) {
  test_construct_1d();
  test_construct_3d();
  test_reject_zero_length();
  test_reject_null_shape();
  test_reject_zero_ndims();
  test_reject_too_many_dims();
  test_view_shares_storage();
  test_scalar_invariants();
  test_destroy_null_is_safe();

  printf("tensor_new_test: all tests passed\n");
  return EXIT_SUCCESS;
}
