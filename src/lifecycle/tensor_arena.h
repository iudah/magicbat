#ifndef TARENA_H
#define TARENA_H

#include <type_alias.h>

typedef struct tarena *tarena;

tarena tarena_new(u64 size);
mem tarena_allocate(tarena arena, u64 size);
void tarena_reset(tarena arena);
void tarena_destroy(tarena arena);

#endif
