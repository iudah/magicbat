#ifndef TENSOR_MEMORY_H
#define TENSOR_MEMORY_H

#include <stdbool.h>
#include <string.h>

#include "tensor_arena.h"
#include "type_alias.h"

mem mballoc(u32 size);
void mbfree(mem ptr);
u64 mbsize(mem ptr);

extern tarena active_arena;
void set_active_arena(tarena arena);

static inline mem tmalloc(u64 size) {
#if defined(USE_MGRIND)
  return mballoc(size);
#else
  return tarena_allocate(active_arena, size);
#endif
}
static inline mem tcalloc(u64 count, u64 size) {
  u64 total = count * size;
  mem ptr = tmalloc(total);

  if (ptr) {
    memset(ptr, 0, total);
  }
  return ptr;
}
static inline mem trealloc(mem ptr, u64 size) {
  if (!ptr)
    return tmalloc(size);

#if defined(USE_MGRIND)
  u64 old_size = mbsize(ptr);
  mem new_ptr = tmalloc(size);

  if (new_ptr) {
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
  }
#else
  mem new_ptr = tmalloc(size);
  // Hack to get a previous size. Too large is irrelevant since we are not
  // zeroing the additional memory
  u64 old_size =
      new_ptr > ptr ? (u8 *)new_ptr - (u8 *)ptr : (u8 *)ptr - (u8 *)new_ptr;

  if (new_ptr) {
    memcpy(new_ptr, ptr, old_size < size ? old_size : size);
  }
#endif
  return new_ptr;
}
static inline bool tfree(mem addr) {
#if defined(USE_MGRIND)
  mbfree(addr);
#else
  (void)addr;
#endif
  return true;
}

#endif
