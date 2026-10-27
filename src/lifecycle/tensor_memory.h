#ifndef TENSOR_MEMORY_H
#define TENSOR_MEMORY_H

#include <stdbool.h>
#include <string.h>
#if defined(ARENA_USE_MALLOC)
#include <stdlib.h>
#endif

#include "tensor_arena.h"
#include "type_alias.h"

mem  mballoc(u32 size);
void mbfree(mem ptr);
u64  mbsize(mem ptr);

extern tarena active_arena;
void          set_active_arena(tarena arena);

#if defined(USE_MGRIND) && defined(ARENA_USE_MALLOC)
#error "USE_MGRIND and ARENA_USE_MALLOC are mutually exclusive"
#endif

static inline mem tmalloc(u64 size) {
#if defined(USE_MGRIND)
  return mballoc(size);
#elif defined(ARENA_USE_MALLOC)
  return malloc(size);
#else
  return tarena_allocate(active_arena, size);
#endif
}

static inline mem tcalloc(u64 count, u64 size) {
#if defined(ARENA_USE_MALLOC)
  return calloc(count, size);
#else
  u64 total = count * size;
  mem ptr   = tmalloc(total);
  if (ptr) {
    memset(ptr, 0, total);
  }
  return ptr;
#endif
}

static inline mem trealloc(mem ptr, u64 size) {
#if defined(ARENA_USE_MALLOC)
  return realloc(ptr, size);
#else
  if (!ptr)
    return tmalloc(size);

#if defined(USE_MGRIND)
  u64 old_size = mbsize(ptr);
  mem new_ptr  = tmalloc(size);
  if (new_ptr) {
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
  }
#else
  mem new_ptr = tmalloc(size);
  u64 old_size =
      new_ptr > ptr ? (u8 *)new_ptr - (u8 *)ptr : (u8 *)ptr - (u8 *)new_ptr;
  if (new_ptr) {
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
  }
#endif
  return new_ptr;
#endif
}

static inline bool tfree(mem addr) {
#if defined(USE_MGRIND)
  mbfree(addr);
#elif defined(ARENA_USE_MALLOC)
  free(addr);
#else
  (void)addr;
#endif
  return true;
}

#endif
