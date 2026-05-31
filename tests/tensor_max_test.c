#include "../include/adt/tensor/tensor_prot.h"
#include "../include/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPS 1e-6

int main(void) {
  printf("Testing max_axis and max_all...\n");

  /* ---------------- tensor_max_axis (axis = 1, along rows) ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){3, 4});

    // Matrix:
    // Row 0: 1  5  2  8
    // Row 1: 9  3  7  4
    // Row 2: 6  2 10  1
    float values[] = {1, 5, 2, 8, 9, 3, 7, 4, 6, 2, 10, 1};
    for (u32 i = 0; i < 12; ++i) {
      t->data->data[i] = values[i];
    }

    Tensor maxed = tensor_max_axis(t, 1); // max along last axis (per row)
    assert(maxed != nullptr);

    // Expected: [8, 9, 10]  (one value per row, keepdim style)
    assert(fabsf(maxed->data->data[0] - 8.0f) < EPS);
    assert(fabsf(maxed->data->data[1] - 9.0f) < EPS);
    assert(fabsf(maxed->data->data[2] - 10.0f) < EPS);

    // Check shape: should be (3, 1)
    assert(tensor_ndims(maxed) == 2);
    assert(tensor_shape(maxed)[0] == 3);
    assert(tensor_shape(maxed)[1] == 1);

    tensor_destroy(maxed);
    tensor_destroy(t);
  }

  /* ---------------- tensor_max_axis (axis = 0, along columns) ----------------
   */
  {
    Tensor t = tensor_new(2, (u32[]){3, 2});

    // Column 0: 1, 9, 6
    // Column 1: 5, 3, 2
    t->data->data[0] = 1;
    t->data->data[1] = 5;
    t->data->data[2] = 9;
    t->data->data[3] = 3;
    t->data->data[4] = 6;
    t->data->data[5] = 2;

    Tensor maxed = tensor_max_axis(t, 0); // max down each column
    assert(maxed != nullptr);

    // Expected: [9, 5]
    assert(fabsf(maxed->data->data[0] - 9.0f) < EPS);
    assert(fabsf(maxed->data->data[1] - 5.0f) < EPS);

    // Shape should be (1, 2)
    assert(tensor_ndims(maxed) == 2);
    assert(tensor_shape(maxed)[0] == 1);
    assert(tensor_shape(maxed)[1] == 2);

    tensor_destroy(maxed);
    tensor_destroy(t);
  }

  /* ---------------- tensor_max_all ---------------- */
  {
    Tensor t = tensor_new(2, (u32[]){2, 3});
    float values[] = {1, 5, 2, 9, 3, 7};
    for (u32 i = 0; i < 6; ++i) {
      t->data->data[i] = values[i];
    }

    float global_max = tensor_max_all(t);
    assert(fabsf(global_max - 9.0f) < EPS);

    tensor_destroy(t);
  }

  /* ---------------- Edge cases ---------------- */
  {
    // Single element
    Tensor single = tensor_new(1, (u32[]){1});
    single->data->data[0] = 42.0f;
    assert(fabsf(tensor_max_all(single) - 42.0f) < EPS);

    Tensor max_single = tensor_max_axis(single, 0);
    assert(max_single != nullptr);
    assert(fabsf(max_single->data->data[0] - 42.0f) < EPS);

    tensor_destroy(max_single);
    tensor_destroy(single);

    // All negative numbers
    Tensor neg = tensor_new(2, (u32[]){2, 2});
    tensor_fill(neg, -10.0f);
    neg->data->data[2] = -3.0f; // highest value

    Tensor max_neg = tensor_max_axis(neg, 1);
    assert(fabsf(max_neg->data->data[0] - (-10.0f)) < EPS);
    assert(fabsf(max_neg->data->data[1] - (-3.0f)) < EPS);

    tensor_destroy(max_neg);
    tensor_destroy(neg);
  }

  printf("All max tests passed!\n");
  return 0;
}
