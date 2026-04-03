#include "../include/tensor/adt/tensor_prot.h"
#include "../include/tensor/tensor.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    printf("Testing broadcasting...\n");

    /* ---------------- (3, 1) + (3, 4) ---------------- */
    {
        Tensor a = tensor_new(2, (u32[]){3, 1});
        Tensor b = tensor_new(2, (u32[]){3, 4});

        assert(a && b);

        tensor_fill(a, 10.0f);           // column vector [10, 10, 10]^T
        tensor_fill(b, 1.0f);            // matrix of ones

        // set different values to verify correctness
        tensor_set(a, (u32[]){1, 0}, 100.0f);   // middle row becomes 100

        Tensor res = tensor_add(a, b);
        assert(res != NULL);

        // Expected:
        // Row 0: 11 11 11 11
        // Row 1: 101 101 101 101
        // Row 2: 11 11 11 11

        assert(res->data[0]  == 11.0f);
        assert(res->data[4]  == 101.0f);   // start of second row
        assert(res->data[7]  == 101.0f);
        assert(res->data[11] == 11.0f);

        tensor_destroy(res);
        tensor_destroy(b);
        tensor_destroy(a);
    }

    /* ---------------- Scalar + Matrix ---------------- */
    {
        Tensor mat = tensor_new(2, (u32[]){2, 3});
        Tensor scalar = tensor_new(1, (u32[]){1});

        assert(mat && scalar);

        tensor_fill(mat, 5.0f);
        scalar->data[0] = 3.0f;

        Tensor res = tensor_mul(mat, scalar);
        assert(res != NULL);

        for (u32 i = 0; i < tensor_num_elements(res); ++i) {
            assert(res->data[i] == 15.0f);
        }

        tensor_destroy(res);
        tensor_destroy(scalar);
        tensor_destroy(mat);
    }

    /* ---------------- Incompatible shapes ---------------- */
    {
        Tensor a = tensor_new(2, (u32[]){2, 3});
        Tensor b = tensor_new(2, (u32[]){4, 2});

        Tensor res = tensor_add(a, b);
        assert(res == NULL);

        tensor_destroy(a);
        tensor_destroy(b);
    }

    printf("All broadcasting tests passed!\n");
    return 0;
}