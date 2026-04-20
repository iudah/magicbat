#ifndef VAR_H
#define VAR_H
#include "../tensor.h"

typedef struct var *Var;

Var track(Tensor t);
bool var_destroy(Var v);
bool var_require_grad(Var v);

Var var_add(Tensor t, Tensor s);
Var var_matmul(Tensor t, Tensor s);

#endif
