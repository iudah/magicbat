#ifndef VAR_PROT_H
#define VAR_PROT_H

#include "../../var/var.h"
#include "../tensor/tensor_prot.h"

struct op {
  void (*backward)(Var self);
  void (*destroy_ctx)(mem ctx);
};

struct var {
  struct tensor_struct base;
  struct var *grad;
  struct var *parent[2];
  struct op op;
  mem ctx;
};

#endif
