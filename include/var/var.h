#ifndef VAR_H
#define VAR_H
#include "../tensor.h"

typedef struct var *Var;

Var track(Tensor t);
bool var_destroy(Var v);

#endif
