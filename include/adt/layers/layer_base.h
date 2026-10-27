#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include "layer_bin.h"
#include "tensor.h"
#include "type_alias.h"
#include <stdio.h>

typedef struct layer_vtable LayerVTable;

struct layer_vtable {
  char             *name;
  u32               n_kernels;
  LayerSerializerId serializer_id;

  Tensor (*forward)(mem self, Tensor input);
  bool   (*deinit)(mem self);
  bool   (*destroy)(mem self);
  u32    (*kernels)(mem self, Tensor buffer[], u32 buffer_length);
  bool   (*track)(mem self);
  bool   (*untrack)(mem self);
};

typedef struct {
  const LayerVTable *vtable;
} LayerBase;

#define LAYER_VT(layer) (*(const LayerVTable **)(layer))

static inline bool layer_vt_valid(mem self) {
  const LayerVTable *virt_table = *(const LayerVTable **)self;
  return virt_table && virt_table->name && virt_table->forward &&
         virt_table->destroy && virt_table->track && virt_table->untrack &&
         virt_table->kernels;
}
static Tensor layer_forward(mem self, Tensor input) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->forward(self, input);
}
static bool layer_deinit(mem self) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->deinit(self);
}
static bool layer_destroy(mem self) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->destroy(self);
}
static u32 layer_kernels(mem self, Tensor buffer[], u32 buffer_length) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->kernels(self, buffer, buffer_length);
}
static u32 layer_n_kernels(mem self) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->n_kernels;
}
static bool layer_track(mem self) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->track(self);
}
static bool layer_untrack(mem self) {
  TASSERT(layer_vt_valid(self) && "layer has no vtable");
  return LAYER_VT(self)->untrack(self);
}
static u32 layers_collect_kernels(mem layers[], u32 n, Tensor *out, u32 limit) {
  u32 kernel_count = 0;
  for (u32 i = 0; i < n; ++i) {
    kernel_count +=
        layer_kernels(layers[i], out + kernel_count,
                      limit > kernel_count ? limit - kernel_count : 0);
  }
  return kernel_count;
}
static void layers_track(mem layers[], u32 n) {
  for (u32 i = 0; i < n; ++i)
    layer_track(layers[i]);
}
static void layers_untrack(mem layers[], u32 n) {
  for (u32 i = 0; i < n; ++i)
    layer_untrack(layers[i]);
}
static void layers_destroy(mem layers[], u32 n) {
  for (u32 i = 0; i < n; ++i)
    layer_destroy(layers[i]);
}
static void layers_write(mem layers[], u32 n, FILE *binary) {
  for (u32 i = 0; i < n; ++i) {
    TASSERT(layer_vt_valid(layers[i]) && "layer has no vtable");
    layer_write(LAYER_VT(layers[i])->serializer_id, layers[i], binary);
  }
}
static void layers_read(mem layers[], u32 n, FILE *binary) {
  for (u32 i = 0; i < n; ++i) {
    layers[i] = layer_read(binary);
  }
}
static Tensor layers_forward(mem layers[], u32 n, Tensor input) {
  auto res = input;
  for (u32 i = 0; i < n; ++i)
    res = layer_forward(layers[i], res);
  return res;
}

#define WRAP_LAYER_CB(base, T)                                                 \
  static Tensor base##_forward_v(mem self, Tensor input) {                     \
    return base##_forward((T)self, input);                                     \
  }                                                                            \
  static bool base##_destroy_v(mem self) { return base##_destroy((T)self); }   \
  static bool base##_track_v(mem self) { return base##_track((T)self); }       \
  static bool base##_untrack_v(mem self) { return base##_untrack((T)self); }   \
  static u32  base##_kernels_v(mem self, Tensor buff[], u32 buffer_length) {   \
    return base##_kernels((T)self, buff, buffer_length);                      \
  }

#define DEFINE_VTABLE(prefix, base, T, name_str, n_k, ser_id)                  \
  WRAP_LAYER_CB(base, T)                                                       \
  static const LayerVTable prefix##_vtable = {                                 \
      .name          = name_str,                                               \
      .n_kernels     = n_k,                                                    \
      .serializer_id = ser_id,                                                 \
      .forward       = base##_forward_v,                                       \
      .destroy       = base##_destroy_v,                                       \
      .track         = base##_track_v,                                         \
      .untrack       = base##_untrack_v,                                       \
      .kernels       = base##_kernels_v,                                       \
  };

#define LAYER_SET_VT(layer, prefix)                                            \
  ((*(const LayerVTable **)(layer)) = &prefix##_vtable)
#endif
