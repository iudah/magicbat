#include "../include/adt/tensor/tensor_prot.h"
#include "../include/tensor.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

#define EPS 1e-6

int main(void) {
    printf("Testing softmax_axis...\n");

    /* ---------------- Softmax along axis 1 (rows) ---------------- */
    {
        Tensor t = tensor_new(2, (u32[]){2, 3});

        // Input:
        // Row 0: 1.0  2.0  3.0
        // Row 1: 0.0  0.0  0.0
        t->data[0] = 1.0f;
        t->data[1] = 2.0f;
        t->data[2] = 3.0f;
        t->data[3] = 0.0f;
        t->data[4] = 0.0f;
        t->data[5] = 0.0f;

        Tensor softmaxed = tensor_softmax_axis(t, 1);  // softmax along last axis (columns)
        assert(softmaxed != NULL);

        // Check that each row sums to \~1.0
        // Row 0
        float row0_sum = softmaxed->data[0] + softmaxed->data[1] + softmaxed->data[2];
        assert(fabsf(row0_sum - 1.0f) < EPS);

        // Row 1 (all zeros → uniform distribution)
        float row1_sum = softmaxed->data[3] + softmaxed->data[4] + softmaxed->data[5];
        assert(fabsf(row1_sum - 1.0f) < EPS);

        // Check that row 0 has increasing probabilities (because 1 < 2 < 3)
        assert(softmaxed->data[0] < softmaxed->data[1]);
        assert(softmaxed->data[1] < softmaxed->data[2]);

        tensor_destroy(softmaxed);
        tensor_destroy(t);
    }

    /* ---------------- Softmax along axis 0 (columns) ---------------- */
    {
        Tensor t = tensor_new(2, (u32[]){3, 2});

        // Column 0: 1, 2, 3
        // Column 1: 0, 0, 0
        t->data[0] = 1.0f;  // row0 col0
        t->data[1] = 0.0f;  // row0 col1
        t->data[2] = 2.0f;  // row1 col0
        t->data[3] = 0.0f;
        t->data[4] = 3.0f;  // row2 col0
        t->data[5] = 0.0f;

        Tensor softmaxed = tensor_softmax_axis(t, 0);  // softmax down each column
        assert(softmaxed != NULL);

        // Each column should sum to \~1.0
        float col0_sum = softmaxed->data[0] + softmaxed->data[2] + softmaxed->data[4];
        float col1_sum = softmaxed->data[1] + softmaxed->data[3] + softmaxed->data[5];

        assert(fabsf(col0_sum - 1.0f) < EPS);
        assert(fabsf(col1_sum - 1.0f) < EPS);

        tensor_destroy(softmaxed);
        tensor_destroy(t);
    }

    /* ---------------- Single element edge case ---------------- */
    {
        Tensor t = tensor_new(1, (u32[]){1});
        t->data[0] = 42.0f;

        Tensor res = tensor_softmax_axis(t, 0);
        assert(res != NULL);
        assert(fabsf(res->data[0] - 1.0f) < EPS);

        tensor_destroy(res);
        tensor_destroy(t);
    }

    printf("All softmax_axis tests passed!\n");
    return 0;
}