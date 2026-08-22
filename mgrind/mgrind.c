#include <assert.h>
#include <backtrace.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <limits.h>
#include <mach-o/dyld.h>
#endif

#define ptrsize (sizeof(uint64_t))
#define one_gigabyte (1024 * 1024 * 1024)
#define trace_buffer_limit (1024)

typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;
typedef int64_t i64;
typedef float f32;
typedef double f64;
typedef void *mem;

struct mblk {
  u64 size;
  char mem[ptrsize];
};
struct mfblk {
  u64 size;
  struct mfblk *next;
};
struct mbpair {
  mem block;
  char *stacktrace;
  struct mbpair *next_pair;
};

typedef struct mbpair mbpair;
typedef struct mblk mblk;

mem arena = nullptr;
mem current_break = nullptr;
u64 available_size = one_gigabyte;
struct backtrace_state *bt_state = nullptr;
char *progname = nullptr;

struct mfblk *freelist = nullptr;
struct mbpair *alloc_list = nullptr;
struct mbpair *freepairs = nullptr;

pthread_mutex_t pairs_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t alloclist_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t freelist_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t arena_lock = PTHREAD_MUTEX_INITIALIZER;

thread_local char trace[trace_buffer_limit];
thread_local u64 trace_buffer_length = 0;

void get_program_path(char *buffer, size_t size);
static u64 align(u32 size) { return (size + (ptrsize - 1)) & ~(ptrsize - 1); }

int bt_callback(void *__attribute((unused)) data,
                uintptr_t __attribute__((unused)) progcounter,
                const char *filename, int lineno, const char *function) {
  int written = snprintf(trace + trace_buffer_length,
                         trace_buffer_limit - trace_buffer_length,
                         "%s:%d in function %s\n", filename, lineno, function);

  if (written > 0) {
    trace_buffer_length += written;
    if (trace_buffer_length >= trace_buffer_limit) {
      trace_buffer_length = trace_buffer_limit - 1;
      trace[trace_buffer_limit - 1] = '\0';
    }
  }
  return 0;
}

void bt_error_callback(void *__attribute__((unused)) data, const char *msg,
                       int errnum) {
  printf("Error %d occurred when getting the stacktrace: %s", errnum, msg);
}

void bt_error_callback_create(void *__attribute__((unused)) data,
                              const char *msg, int errnum) {
  printf("Error %d occurred when initializing the stacktrace: %s", errnum, msg);
}

void init_back_trace(const char *filename) {
  bt_state =
      backtrace_create_state(filename, 0, bt_error_callback_create, nullptr);
}

void get_back_trace() {
  if (!bt_state) {
    printf("Make sure init_back_trace() is called before calling "
           "print_stack_trace()\n");
    abort();
  }

  *trace = 0;
  trace_buffer_length = 0;

  backtrace_full(bt_state, 0, bt_callback, bt_error_callback, nullptr);
}

struct mbpair *new_pair() {
  pthread_mutex_lock(&pairs_lock);
  auto pair = freepairs;
  if (pair) {
    freepairs = pair->next_pair;
  }
  pthread_mutex_unlock(&pairs_lock);
  if (!pair)
    return malloc(sizeof(struct mbpair));
  return pair;
}

void map_append(mem block, char *stacktrace) {
  struct mbpair *pair = new_pair();
  pthread_mutex_lock(&alloclist_lock);
  pair->next_pair = alloc_list;
  alloc_list = pair;
  alloc_list->block = block;
  alloc_list->stacktrace = stacktrace;
  pthread_mutex_unlock(&alloclist_lock);
}

void drop_map(mem block) {
  pthread_mutex_lock(&alloclist_lock);
  pthread_mutex_lock(&pairs_lock);

  struct mbpair *prev = nullptr;
  struct mbpair *current = alloc_list;
  while (current) {
    if (current->block == block) {
      // Remove from alloc_list
      if (prev)
        prev->next_pair = current->next_pair;
      else
        alloc_list = current->next_pair;

      // Recycle the pair
      current->next_pair = freepairs;
      freepairs = current;
      free(current->stacktrace);

      pthread_mutex_unlock(&pairs_lock);
      pthread_mutex_unlock(&alloclist_lock);
      return;
    }
    prev = current;
    current = current->next_pair;
  }

  pthread_mutex_unlock(&pairs_lock);
  pthread_mutex_unlock(&alloclist_lock);
}

u64 mbsize(mem ptr) {
  struct mblk *blk = (struct mblk *)((u8 *)ptr - offsetof(struct mblk, mem));
  return blk->size - offsetof(struct mblk, mem);
}

struct mblk *new_block(u32 size) {
  size = align(size + offsetof(struct mblk, mem));
  pthread_mutex_lock(&arena_lock);
  if (available_size < size) {
    pthread_mutex_unlock(&arena_lock);
    return nullptr;
  }
  available_size -= size;
  u8 *brk = (u8 *)current_break;
  current_break = brk + size;
  pthread_mutex_unlock(&arena_lock);
  auto blk = (struct mblk *)brk;
  blk->size = size;
  return blk;
}

struct mblk *bestfree(u32 size) {
  pthread_mutex_lock(&freelist_lock);
  u64 bestsize = UINT64_MAX;
  struct mfblk *best = nullptr;
  struct mfblk *prevbest = nullptr;
  struct mfblk *prev = nullptr;
  struct mfblk *current = freelist;

  while (current) {
    if (current->size >= size && current->size < bestsize) {
      bestsize = current->size;
      best = current;
      prevbest = prev;
    }
    prev = current;
    current = current->next;
  }

  if (best) {
    if (prevbest) {
      prevbest->next = best->next;
    } else {
      freelist = best->next;
    }
  }

  pthread_mutex_unlock(&freelist_lock);

  return (struct mblk *)best;
}

mem mballoc(u32 size) {
  size = align(size);
  struct mblk *ptr = bestfree(size);
  if (!ptr) {
    ptr = new_block(size);
    if (!ptr)
      return nullptr;
  }
  get_back_trace();
  map_append(ptr, strdup(trace));

  return ptr->mem;
}

void mbfree(mem ptr) {
  if (!ptr)
    return;
  struct mblk *blk = (struct mblk *)((u8 *)ptr - offsetof(struct mblk, mem));
  struct mfblk *freeblk = (struct mfblk *)blk;

  pthread_mutex_lock(&freelist_lock);
  freeblk->size = blk->size;
  freeblk->next = freelist;
  freelist = freeblk;
  pthread_mutex_unlock(&freelist_lock);

  drop_map(blk);
}

void get_program_path(char *buffer, size_t size) {
#if defined(_WIN32)
  GetModuleFileName(NULL, buffer, (DWORD)size);
#elif defined(__linux__)
  ssize_t len = readlink("/proc/self/exe", buffer, size - 1);
  if (len != -1)
    buffer[len] = '\0';
#elif defined(__APPLE__)
  uint32_t bufsize = (uint32_t)size;
  _NSGetExecutablePath(buffer, &bufsize);
#else
  snprintf(buffer, size, "Unsupported platform");
#endif
}

static void __attribute__((constructor)) init() {
  // allocate arena
  arena = current_break = malloc(one_gigabyte);
  if (!arena)
    abort();
#define buffer_path (2048)
  char path[buffer_path];
  get_program_path(path, sizeof(path));
  init_back_trace(path);
  printf("%s\n\n", path);
}

static void __attribute__((destructor)) deinit() {
  auto current = alloc_list;
  while (current) {
    printf("Location: %p\nTrace: \n%s\n\n",
           ((struct mblk *)current->block)->mem, current->stacktrace);
    current = current->next_pair;
  }
  free(arena);
}
