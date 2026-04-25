#ifndef VAR_PROT_H
#define VAR_PROT_H

#include "../../var/var.h"
#include "../tensor/tensor_prot.h"

typedef struct op {
  void (*backward)(Var self);
  void (*destroy_ctx)(mem ctx);
} VarOp;

struct var_struct {
  struct tensor_struct base;
  struct tensor_struct *grad;
  struct var_struct *parent[2];
  struct op op;
  mem ctx;
  u64 mark;
};

static inline void var_track_parent(Tensor t, Tensor a, Tensor b, VarOp op,
                                    void *ctx) {
  Var res = (Var)t;
  res->parent[0] = (Var)a;
  res->parent[1] = (Var)b;

  if (a)
    atomic_fetch_add(&a->refcount, 1);
  if (b)
    atomic_fetch_add(&b->refcount, 1);

  res->op = op;
  res->ctx = ctx;
}

#endif
