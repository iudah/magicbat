#include "tensor_arena.h"
#include <stdlib.h>
#include <type_alias.h>

struct tarena {
  mem buffer;
  u64 size;
  u64 available_size;
  mem brk;
};

#define ptrsize (sizeof(uint64_t))
static u64 align(u32 size) { return (size + (ptrsize - 1)) & ~(ptrsize - 1); }

tarena tarena_new(u64 size) {
#if defined(ARENA_USE_MALLOC)
  (void)size;
  return (tarena)1;
#else
  tarena arena          = malloc(sizeof(*arena));
  size                  = align(size);
  arena->size           = size;
  arena->available_size = size;
  arena->buffer = arena->brk = malloc(size);
  return arena;
#endif
}

mem tarena_allocate(tarena arena, u64 size) {
#if defined(ARENA_USE_MALLOC)
  (void)arena;
  (void)size;
  abort();
#else
  size = align(size);
  if (size == 0)
    return nullptr;
  if (arena->available_size < size) {
    fprintf(stderr, "Arena out of memory.\n");
    return nullptr;
  }
  auto allocation = arena->brk;
  arena->brk      = (char *)arena->brk + size;
  arena->available_size -= size;
  return allocation;
#endif
}

void tarena_reset(tarena arena) {
#if defined(ARENA_USE_MALLOC)
  (void)arena;
#else
  arena->available_size = arena->size;
  arena->brk            = arena->buffer;
#endif
}

void tarena_destroy(tarena arena) {
#if defined(ARENA_USE_MALLOC)
  (void)arena;
#else
  free(arena->buffer);
  free(arena);
#endif
}
