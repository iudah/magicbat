#include "tensor_memory.h"
#include "tensor_arena.h"

tarena active_arena = nullptr;
void set_active_arena(tarena arena) { active_arena = arena; }
