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

static inline void var_track_parent(Tensor tensor, Tensor parent_a,
                                    Tensor parent_b, VarOp operation,
                                    void *ctx) {
  Var res = (Var)tensor;
  res->parent[0] = (Var)parent_a;
  res->parent[1] = (Var)parent_b;

  if (parent_a)
    atomic_fetch_add(&parent_a->refcount, 1);
  if (parent_b)
    atomic_fetch_add(&parent_b->refcount, 1);

  res->op = operation;
  res->ctx = ctx;
}

#endif
