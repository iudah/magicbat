#ifndef VAR_H
#define VAR_H
#include "../tensor.h"

typedef struct var_struct *Var;

Tensor track(Tensor t);
void untrack(Tensor t);
bool var_destroy(Tensor v);
bool var_require_grad(Tensor v);
bool var_is_tensor(Tensor v);
Tensor var_grad(Tensor v);

Tensor var_add(Tensor var_a, Tensor var_b);
Tensor var_sub(Tensor t, Tensor s);
Tensor var_mul(Tensor t, Tensor s);
Tensor var_div(Tensor t, Tensor s);
Tensor var_matmul(Tensor t, Tensor s);

Tensor var_relu(Tensor a);
Tensor var_sum_all(Tensor a);

bool var_backward(Tensor top);
bool var_backward_with_grad(Tensor top, Tensor grad);
bool var_backward_verbose(Tensor top, bool user_thread_safety_assured,
                          Tensor grad);
bool var_zero_grad(Tensor top);

#endif
