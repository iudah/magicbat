#include "../../include/adt/var/var_prot.h"
#include "../../include/tensor.h"
#include "../lifecycle/tensor_memory.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

typedef struct {
  Var *list;
  u32 cap;
  u32 count;
} sorted_list;
typedef struct {
  Var v;
  enum { UNDEFINED, ENTRY, EXIT } flag;
} stack_item;
typedef struct {
  stack_item *stack;
  u32 cap;
  u32 count;
} sort_stack;

#define CAPACITY (32)

bool list_append(sorted_list *list, Var var) {
  if (!list->list) {
    list->cap = CAPACITY;
    list->list = tmalloc(list->cap * sizeof(*list->list));
  }
  if (list->count == list->cap) {
    u32 new_cap = list->count * 2;
    mem tmp = trealloc(list->list, new_cap * sizeof(*list->list));
    if (!tmp)
      return false;
    list->list = tmp;
    list->cap = new_cap;
  }
  if (!list->list)
    return false;

  list->list[list->count] = var;
  ++list->count;
  return true;
}

static inline bool stack_push(sort_stack *stack, stack_item item) {
  if (!stack->stack) {
    stack->cap = CAPACITY;
    stack->stack = tmalloc(stack->cap * sizeof(*stack->stack));
  }
  if (stack->count == stack->cap) {
    u32 new_cap = stack->count * 2;
    mem tmp = trealloc(stack->stack, new_cap * sizeof(*stack->stack));
    if (!tmp)
      return false;
    stack->stack = tmp;
    stack->cap = new_cap;
  }
  if (!stack->stack)
    return false;

  stack->stack[stack->count] = item;
  ++stack->count;

  return true;
}

static inline stack_item stack_pop(sort_stack *stack) {
  if (!stack->count)
    return (stack_item){nullptr, 0};
  return stack->stack[--stack->count];
}

static inline bool stack_is_empty(sort_stack *stack) {
  return stack->count == 0;
}

_Atomic static u64 global_mark = 0;

static inline sorted_list topological_sort(Var top) {
  sorted_list list = {nullptr, 0, 0};
  sort_stack stack = {nullptr, 0, 0};

  u64 current_mark = atomic_fetch_add(&global_mark, 1) + 1;

  if (!stack_push(&stack, (stack_item){top, ENTRY})) {
    tfree(stack.stack);
    tfree(list.list);
    return (sorted_list){nullptr, 0, 0};
  }

  while (!stack_is_empty(&stack)) {
    stack_item item = stack_pop(&stack);

    //    if (item.flag == UNDEFINED) {
    //    break;
    //}

    Var variable = item.v;
    if (item.flag == EXIT) {
      list_append(&list, variable);
      if (variable->grad)
        tensor_fill(variable->grad, 0);
      continue;
    }

    if (variable->mark == current_mark)
      continue;

    variable->mark = current_mark;

    if (item.flag == ENTRY && variable->base.requires_grad &&
        !variable->base.is_tensor_type) {
      stack_push(&stack, (stack_item){variable, EXIT});
      if (variable->parent[0] && !variable->parent[0]->base.is_tensor_type &&
          variable->parent[0]->base.requires_grad)
        stack_push(&stack, (stack_item){variable->parent[0], ENTRY});
      if (variable->parent[1] && !variable->parent[1]->base.is_tensor_type &&
          variable->parent[1]->base.requires_grad)
        stack_push(&stack, (stack_item){variable->parent[1], ENTRY});
    }
  }

  tfree(stack.stack);

  return list;
}

pthread_mutex_t topo_mutex = PTHREAD_MUTEX_INITIALIZER;

bool var_backward_verbose(Tensor top, bool user_thread_safety_assured,
                          Tensor grad) {
  Var var = (Var)top;

  if (!top || top->is_tensor_type || !top->requires_grad)
    return false;

  if (!user_thread_safety_assured)
    pthread_mutex_lock(&topo_mutex);

  bool ret_val = true;
  sorted_list list = topological_sort(var);
  if (!list.list) {
    ret_val = false;
    goto skip_backward;
  }

  if (grad)
    var->grad = grad;
  else {
    if (!var->grad)
      var->grad = tensor_new(var->base.ndims, var->base.shape);

    tensor_fill(var->grad, 1.0F);
  }

  for (u32 i = list.count; i-- > 0;) {
    Var variable = list.list[i];
    if (variable->op.backward)
      variable->op.backward(variable);
  }

skip_backward:
  if (!user_thread_safety_assured)
    pthread_mutex_unlock(&topo_mutex);

  tfree(list.list);

  return ret_val;
}

bool var_backward(Tensor top) {
  return var_backward_verbose(top, false, nullptr);
}
bool var_backward_with_grad(Tensor top, Tensor grad) {
  return var_backward_verbose(top, false, grad);
}
Tensor var_grad(Tensor var) {
  Var variable = (Var)var;
  return variable->grad;
}
